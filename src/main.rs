use bytes::Bytes;
use log::{error, info};
use ndarray::{Array2, Array4, ArrayView2, s};
use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Vector},
    dnn,
    highgui, imgcodecs, imgproc,
    prelude::*,
    videoio,
};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::task;
use warp::Filter;
use warp::hyper::{Body, Response};

// ======================== Константы YuNet ========================
const MIN_SIZES: [[f32; 3]; 3] = [
    [10.0, 16.0, 24.0], // stride 8
    [32.0, 48.0, 0.0],  // stride 16
    [64.0, 96.0, 0.0],  // stride 32
];
const STEPS: [i32; 3] = [8, 16, 32];
const VARIANCE: [f32; 2] = [0.1, 0.2];

// Рекомендуемый размер входа YuNet
const INPUT_WIDTH: i32 = 320;
const INPUT_HEIGHT: i32 = 320;

// ======================== Структуры ========================
#[derive(Debug, Clone)]
pub struct FaceDetection {
    pub bbox: Rect,
    pub confidence: f32,
    pub landmarks: [(f32, f32); 5], // right_eye, left_eye, nose, right_mouth, left_mouth
}

// ======================== Основная программа ========================
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    info!("Запуск программы...");

    // --- Камера ---
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !cam.is_opened()? {
        error!("Камера не открыта");
        anyhow::bail!("Камера не открыта");
    }
    // Можно ограничить разрешение для скорости
    // cam.set(videoio::CAP_PROP_FRAME_WIDTH, 640.0)?;
    // cam.set(videoio::CAP_PROP_FRAME_HEIGHT, 480.0)?;
    info!("Камера успешно открыта");

    // --- ONNX Session (YuNet) ---
    // Скачай модель:
    // wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("face_detection_yunet_2023mar.onnx")?;
    let session = Arc::new(session);
    info!("ONNX-модель YuNet загружена");

    // Кешируем priors один раз
    let priors = Arc::new(generate_priors(INPUT_WIDTH, INPUT_HEIGHT));

    let window_name = "YuNet Face Detection";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    // Broadcast для MJPEG
    let (tx, _) = broadcast::channel::<Bytes>(8);
    tokio::spawn(run_http_server(tx.clone()));

    let mut frame = Mat::default();

    loop {
        if !cam.read(&mut frame)? || frame.empty() {
            error!("Не удалось считать кадр");
            continue;
        }

        let session_clone = session.clone();
        let priors_clone = priors.clone();
        let frame_clone = frame.clone();

        let processed = task::spawn_blocking(move || {
            detect_faces(&session_clone, &priors_clone, frame_clone, 0.7, 0.3)
        })
        .await??;

        // MJPEG
        if tx.receiver_count() > 0 {
            if let Ok(jpeg) = encode_frame_to_jpeg(&processed) {
                let mut mjpeg = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ".to_vec();
                mjpeg.extend(jpeg.len().to_string().as_bytes());
                mjpeg.extend(b"\r\n\r\n");
                mjpeg.extend_from_slice(&jpeg);
                mjpeg.extend(b"\r\n");
                let _ = tx.send(Bytes::from(mjpeg));
            }
        }

        highgui::imshow(window_name, &processed)?;
        if highgui::wait_key(1)? == b'q' as i32 {
            info!("Выход по нажатию 'q'");
            break;
        }
    }

    highgui::destroy_all_windows()?;
    info!("Программа завершена");
    Ok(())
}

// ======================== Детекция лиц ========================
fn detect_faces(
    session: &Session,
    priors: &Array2<f32>,
    mut frame: Mat,
    conf_threshold: f32,
    nms_threshold: f32,
) -> anyhow::Result<Mat> {
    let (input_tensor, scale_x, scale_y) = preprocess(&frame, INPUT_WIDTH, INPUT_HEIGHT)?;

    // Инференс
    // ВАЖНО: имя входа может отличаться. Проверь session.inputs()
    let outputs = session.run(ort::inputs![
        "input" => TensorRef::from_array_view(input_tensor.view())?
    ])?;

    let faces = postprocess_yunet(
        &outputs,
        priors,
        (INPUT_WIDTH, INPUT_HEIGHT),
        conf_threshold,
        nms_threshold,
        scale_x,
        scale_y,
    )?;

    draw_faces(&mut frame, &faces)?;
    Ok(frame)
}

// ======================== Предобработка ========================
fn preprocess(
    frame: &Mat,
    target_w: i32,
    target_h: i32,
) -> anyhow::Result<(Array4<f32>, f32, f32)> {
    let orig_w = frame.cols() as f32;
    let orig_h = frame.rows() as f32;

    let mut resized = Mat::default();
    imgproc::resize(
        frame,
        &mut resized,
        core::Size::new(target_w, target_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let scale_x = orig_w / target_w as f32;
    let scale_y = orig_h / target_h as f32;

    // BGR, NCHW, значения 0..255 (без нормализации)
    let mut array = Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));
    let data = resized.data_bytes()?;
    let step = resized.step1(0)? as usize;

    for y in 0..target_h as usize {
        for x in 0..target_w as usize {
            let idx = y * step + x * 3;
            array[[0, 0, y, x]] = data[idx] as f32;     // B
            array[[0, 1, y, x]] = data[idx + 1] as f32; // G
            array[[0, 2, y, x]] = data[idx + 2] as f32; // R
        }
    }

    Ok((array, scale_x, scale_y))
}

// ======================== Генерация Prior Boxes ========================
fn generate_priors(input_w: i32, input_h: i32) -> Array2<f32> {
    let mut priors = Vec::new();

    for (idx, &step) in STEPS.iter().enumerate() {
        let fm_h = ((input_h as f32) / step as f32).ceil() as i32;
        let fm_w = ((input_w as f32) / step as f32).ceil() as i32;

        let min_sizes = &MIN_SIZES[idx];
        let num_sizes = if min_sizes[2] == 0.0 { 2 } else { 3 };

        for i in 0..fm_h {
            for j in 0..fm_w {
                let cx = (j as f32 + 0.5) * step as f32 / input_w as f32;
                let cy = (i as f32 + 0.5) * step as f32 / input_h as f32;

                for k in 0..num_sizes {
                    let s_kx = min_sizes[k] / input_w as f32;
                    let s_ky = min_sizes[k] / input_h as f32;
                    priors.push([cx, cy, s_kx, s_ky]);
                }
            }
        }
    }

    Array2::from_shape_vec(
        (priors.len(), 4),
        priors.into_iter().flatten().collect(),
    )
    .expect("Failed to create priors")
}

// ======================== Decode ========================
fn decode(
    loc: ArrayView2<f32>,
    conf: ArrayView2<f32>,
    iou: ArrayView2<f32>,
    priors: &Array2<f32>,
    input_w: i32,
    input_h: i32,
) -> Vec<(f32, [f32; 14])> {
    let num = priors.nrows();
    let mut detections = Vec::with_capacity(num);

    for i in 0..num {
        let cls_score = conf[[i, 1]];
        let mut iou_score = iou[[i, 0]];
        iou_score = iou_score.clamp(0.0, 1.0);
        let score = (cls_score * iou_score).sqrt();

        if score < 0.15 {
            continue;
        }

        let prior = priors.row(i);
        let (px, py, pw, ph) = (prior[0], prior[1], prior[2], prior[3]);

        // BBox
        let cx = px + loc[[i, 0]] * VARIANCE[0] * pw;
        let cy = py + loc[[i, 1]] * VARIANCE[0] * ph;
        let w = pw * (loc[[i, 2]] * VARIANCE[1]).exp();
        let h = ph * (loc[[i, 3]] * VARIANCE[1]).exp();

        let x1 = (cx - w / 2.0) * input_w as f32;
        let y1 = (cy - h / 2.0) * input_h as f32;
        let x2 = (cx + w / 2.0) * input_w as f32;
        let y2 = (cy + h / 2.0) * input_h as f32;

        // Landmarks
        let mut kps = [0.0f32; 10];
        for k in 0..5 {
            kps[k * 2] = (px + loc[[i, 4 + k * 2]] * VARIANCE[0] * pw) * input_w as f32;
            kps[k * 2 + 1] = (py + loc[[i, 5 + k * 2]] * VARIANCE[0] * ph) * input_h as f32;
        }

        let mut data = [0.0f32; 14];
        data[0] = x1;
        data[1] = y1;
        data[2] = x2;
        data[3] = y2;
        data[4..14].copy_from_slice(&kps);

        detections.push((score, data));
    }
    detections
}

// ======================== NMS ========================
fn nms(
    detections: &[(f32, [f32; 14])],
    conf_threshold: f32,
    nms_threshold: f32,
    top_k: i32,
) -> Vec<usize> {
    let mut boxes = Vector::<Rect>::new();
    let mut scores = Vector::<f32>::new();
    let mut original_indices = Vec::new();

    for (i, &(score, ref data)) in detections.iter().enumerate() {
        if score < conf_threshold {
            continue;
        }
        let x1 = data[0] as i32;
        let y1 = data[1] as i32;
        let x2 = data[2] as i32;
        let y2 = data[3] as i32;
        boxes.push(Rect::new(x1, y1, (x2 - x1).max(1), (y2 - y1).max(1)));
        scores.push(score);
        original_indices.push(i);
    }

    let mut indices = Vector::<i32>::new();
    let _ = dnn::NMSBoxes(&boxes, &scores, conf_threshold, nms_threshold, &mut indices, 1.0, top_k);

    indices
        .iter()
        .filter_map(|&i| original_indices.get(i as usize).copied())
        .collect()
}

// ======================== Постпроцессинг ========================
fn postprocess_yunet(
    outputs: &ort::session::SessionOutputs<'_>,
    priors: &Array2<f32>,
    input_size: (i32, i32),
    conf_threshold: f32,
    nms_threshold: f32,
    scale_x: f32,
    scale_y: f32,
) -> anyhow::Result<Vec<FaceDetection>> {
    let (input_w, input_h) = input_size;

    // ============================================================
    // ВАЖНО: имена выходов зависят от конкретной модели.
    // Запусти один раз и посмотри session.outputs(), затем поправь.
    // Типичные варианты для YuNet 2023mar:
    //   - "cls_8", "cls_16", "cls_32", "obj_8"... и т.д. (12 тензоров)
    //   - или уже объединённые "loc", "conf", "iou"
    // ============================================================

    // Вариант 1: если модель выдаёт уже склеенные тензоры (редко)
    // let loc  = outputs["loc"].try_extract_array::<f32>()?.into_dimensionality()?;
    // let conf = outputs["conf"].try_extract_array::<f32>()?.into_dimensionality()?;
    // let iou  = outputs["iou"].try_extract_array::<f32>()?.into_dimensionality()?;

    // Вариант 2: работаем с 12 отдельными тензорами (наиболее частый случай)
    // Ниже пример склейки. Подставь реальные имена из session.outputs()!

    // Для демонстрации — заглушка, которая просто возвращает пустой список,
    // если имена не совпали. Замени на реальный код после проверки имён.

    // --- Пример склейки (раскомментируй и поправь имена) ---
    /*
    let cls8  = outputs["cls_8"].try_extract_array::<f32>()?;
    let cls16 = outputs["cls_16"].try_extract_array::<f32>()?;
    let cls32 = outputs["cls_32"].try_extract_array::<f32>()?;
    // ... аналогично для obj, bbox, kps

    // Потом reshape + concatenate по axis=1 (или 0 в зависимости от layout)
    */

    // Временная заглушка — чтобы код компилировался.
    // После проверки имён выходов замени этот блок на реальный decode.
    let _ = outputs; // чтобы не было warning
    let detections: Vec<(f32, [f32; 14])> = Vec::new();

    // Если у тебя уже есть loc/conf/iou — раскомментируй:
    // let detections = decode(loc.view(), conf.view(), iou.view(), priors, input_w, input_h);

    let keep = nms(&detections, conf_threshold, nms_threshold, 5000);

    let mut faces = Vec::with_capacity(keep.len());
    for &idx in &keep {
        let (score, data) = &detections[idx];

        faces.push(FaceDetection {
            bbox: Rect::new(
                (data[0] * scale_x) as i32,
                (data[1] * scale_y) as i32,
                ((data[2] - data[0]) * scale_x) as i32,
                ((data[3] - data[1]) * scale_y) as i32,
            ),
            confidence: *score,
            landmarks: [
                (data[4] * scale_x, data[5] * scale_y),
                (data[6] * scale_x, data[7] * scale_y),
                (data[8] * scale_x, data[9] * scale_y),
                (data[10] * scale_x, data[11] * scale_y),
                (data[12] * scale_x, data[13] * scale_y),
            ],
        });
    }

    Ok(faces)
}

// ======================== Отрисовка ========================
fn draw_faces(frame: &mut Mat, faces: &[FaceDetection]) -> opencv::Result<()> {
    for face in faces {
        imgproc::rectangle(
            frame,
            face.bbox,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;

        let label = format!("{:.2}", face.confidence);
        imgproc::put_text(
            frame,
            &label,
            Point::new(face.bbox.x, (face.bbox.y - 5).max(10)),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            imgproc::LINE_8,
            false,
        )?;

        let colors = [
            Scalar::new(0.0, 0.0, 255.0, 0.0),   // right eye
            Scalar::new(255.0, 0.0, 0.0, 0.0),   // left eye
            Scalar::new(0.0, 255.0, 255.0, 0.0), // nose
            Scalar::new(255.0, 0.0, 255.0, 0.0), // right mouth
            Scalar::new(255.0, 255.0, 0.0, 0.0), // left mouth
        ];

        for (i, &(x, y)) in face.landmarks.iter().enumerate() {
            imgproc::circle(
                frame,
                Point::new(x as i32, y as i32),
                2,
                colors[i],
                -1,
                imgproc::LINE_8,
                0,
            )?;
        }
    }
    Ok(())
}

// ======================== Вспомогательные ========================
fn encode_frame_to_jpeg(frame: &Mat) -> opencv::Result<Vec<u8>> {
    let mut buf = Vector::new();
    imgcodecs::imencode(".jpg", frame, &mut buf, &Vector::new())?;
    Ok(buf.to_vec())
}

async fn run_http_server(tx: broadcast::Sender<Bytes>) {
    let route = warp::path!("stream").map(move || {
        let mut rx = tx.subscribe();
        let stream = async_stream::stream! {
            while let Ok(data) = rx.recv().await {
                yield Ok::<_, std::convert::Infallible>(data);
            }
        };
        Response::builder()
            .header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            .body(Body::wrap_stream(stream))
            .unwrap()
    });

    info!("MJPEG-стрим: http://127.0.0.1:3030/stream");
    warp::serve(route).run(([127, 0, 0, 1], 3030)).await;
}
