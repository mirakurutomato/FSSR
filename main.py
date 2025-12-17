import cv2
import numpy as np
import time
import mediapipe as mp
import os

# --- Retinex 関連関数 ---
def gaussian_separable_blur(img_gray: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    g = cv2.getGaussianKernel(ksize, sigma)
    blurred = cv2.sepFilter2D(img_gray, -1, g, g)
    return blurred

def fastretinex_stable(rgb: np.ndarray, sigma: float = 15.0, eps: float = 1e-6,
                        clip_scale=(0.5, 1.5), gamma: float = 0.8):
    t0 = time.perf_counter()
    img = rgb.astype(np.float32) / 255.0
    Y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    t1 = time.perf_counter()
    L_sep = gaussian_separable_blur(Y, sigma=sigma)
    t2 = time.perf_counter()

    L = np.clip(L_sep, eps, 1.0)
    log_Y = np.log(Y + eps)
    log_R = log_Y - np.log(L + eps)
    R = np.exp(log_R)

    scale = np.clip(R / (Y + eps), clip_scale[0], clip_scale[1])
    out = img * scale[:, :, None]
    out = np.clip(out, 0.0, 1.0)

    out = np.power(out, gamma)
    out_uint8 = (out * 255.0).astype(np.uint8)

    t3 = time.perf_counter()
    timings = {"sep_time": t2 - t1, "full_time": t3 - t2, "total_time": t3 - t0}

    return out_uint8, L_sep.astype(np.float32), timings

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))

# --- メイン関数 ---
def run_camera_benchmark_with_hands():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return

    sigma = 15.0
    frame_count = 0
    os.makedirs("screenshots", exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           model_complexity=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 255, 0)
    thickness = 1

    t_start = time.perf_counter()
    t_prev = t_start

    # 各フレームの情報を記録するリスト
    fps_list = []
    sep_time_list = []
    full_time_list = []
    mse_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out, L_sep, timings = fastretinex_stable(rgb, sigma=sigma)

        # 手検出（補正前）
        results_before = hands.process(rgb)
        frame_before = frame.copy()
        if results_before.multi_hand_landmarks:
            for hand_landmarks in results_before.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_before, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 手検出（補正後）
        results_after = hands.process(out)
        frame_after = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        if results_after.multi_hand_landmarks:
            for hand_landmarks in results_after.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_after, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # MSE
        gray_out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        frame_mse = mse(rgb[:, :, 0], gray_out)

        # FPS計算
        t_now = time.perf_counter()
        fps = 1.0 / (t_now - t_prev) if (t_now - t_prev) > 0 else 0.0
        t_prev = t_now

        # 情報をリストに追加
        fps_list.append(fps)
        sep_time_list.append(timings['sep_time'])
        full_time_list.append(timings['full_time'])
        mse_list.append(frame_mse)

        frame_count += 1

        # 左右表示（情報文字は表示しない）
        disp = np.hstack((frame_before, frame_after))
        cv2.imshow("FastRetinex + Hands (Before | After)", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            filename = f"screenshots/frame_{frame_count:04d}.png"
            cv2.imwrite(filename, disp)
            print(f"Screenshot saved: {filename}")
        elif key in [ord('+'), ord('=')]:
            sigma = max(1.0, sigma - 2.0)
            print("sigma:", sigma)
        elif key in [ord('-'), ord('_')]:
            sigma += 2.0
            print("sigma:", sigma)

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    # 平均値計算
    avg_fps = np.mean(fps_list) if fps_list else 0
    avg_sep = np.mean(sep_time_list) if sep_time_list else 0
    avg_full = np.mean(full_time_list) if full_time_list else 0
    avg_mse = np.mean(mse_list) if mse_list else 0

    # CUI出力
    print("\n--- CAMERA SESSION SUMMARY ---")
    print(f"Camera runtime: {elapsed:.2f} s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average sep time: {avg_sep*1000:.2f} ms")
    print(f"Average full time: {avg_full*1000:.2f} ms")
    print(f"Average MSE(L): {avg_mse:.6f}")
    print(f"Total frames processed: {frame_count}")

if __name__ == "__main__":
    run_camera_benchmark_with_hands()
