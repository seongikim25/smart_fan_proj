import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import board
import busio
import adafruit_mlx90640
import serial  # 시리얼 통신 라이브러리
import glob
# from gpiozero import DigitalOutputDevice # GPIOzero 제거됨

# ==================================================
# 설정값
# ==================================================
# SSR 핀 (BCM 번호) -> 아두이노가 제어하므로 주석 처리
# SSR_PIN_LOW = 17    # 미풍 (약풍)
# SSR_PIN_MID = 27    # 중풍
# SSR_PIN_HIGH = 26   # 강풍

# 포트 설정
PRIZM_PORT = "/dev/prizm"    # TETRIX PRIZM (회전)
UNO_PORT = "/dev/arduino"  # 아두이노 우노 포트 (예: /dev/ttyACM0)
BAUD = 115200

RGB_CAM_INDEX = 0
MLX_REFRESH = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
PD_TS = 0.5

# 열화상 파라미터
HUMAN_TEMP_MIN = 25.0
HUMAN_TEMP_MAX = 38.0
TEMP_THRESHOLD_L = 30.0
TEMP_THRESHOLD_M = 32.0
TEMP_THRESHOLD_H = 34.0

VIS_INTERVAL = 0.2
GESTURE_STABLE_TIME = 0.15

# ==================================================
# 공용 데이터
# ==================================================
state = {
    "mlx_frame": None,
    "detected_hand_landmarks": None,
    "gesture_status": "NONE",
    "control_mode": "GESTURE",
    "running": True,
    "last_speed_cmd": "S",
    "last_turn_cmd": "S",
    "mode_selecting": False,
    "mode_select_start": None,
    "selected_mode": None,
    "hot_spot_coords": None,
    "target_temp": None,
    "is_rotating": False,
    "is_locked": False,
    "locked_speed": "S",
    "lock_selecting": False,
    "lock_select_start": None,
    "selected_lock_action": None,
    "rotation_selecting": False,
    "rotation_select_start": None,
    "selected_rotation_action": None,
    "last_toggle_rotation_ts": 0.0,
    "last_toggle_lock_ts": 0.0,
    "last_detected_gesture": "NONE",
    "last_detected_gesture_time": 0.0,
    "prizm_connected": False
}
state_lock = threading.Lock()

# ==================================================
# SSR 제어 클래스 (아두이노 우노 버전)
# ==================================================
class SSRController:
    def __init__(self, port, baud):
        """아두이노 우노와 시리얼 연결을 시도합니다."""
        self.ser = None
        self.port = port
        self.baud = baud
        self.is_connected = False
        self.connect()

    def connect(self):
        """지정된 포트에 시리얼 연결 시도"""
        try:
            # 포트가 None이거나 비어있으면 자동 검색 시도
            if not self.port:
                ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
                if not ports:
                    print("[SSR-UNO] 아두이노 포트를 찾을 수 없습니다.")
                    return
                self.port = ports[0] # 찾은 첫 번째 포트 사용
                print(f"[SSR-UNO] 자동 감지된 포트 사용: {self.port}")

            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)  # 아두이노 부트로더 및 재시작 대기
            self.ser.flush()
            self.is_connected = True
            print(f"[SSR-UNO] 연결 성공: {self.port}")
            self.all_off()  # 연결 직후 정지 명령 전송
        except Exception as e:
            print(f"[SSR-UNO] 연결 실패 ({self.port}): {e}")
            self.ser = None
            self.is_connected = False

    def all_off(self):
        """모든 SSR OFF (정지 'S' 명령 전송)"""
        self.set_speed("S")

    def set_speed(self, cmd):
        """명령 (L, M, H, S)을 아두이노로 전송"""
        if not self.is_connected or self.ser is None or not self.ser.is_open:
            # print("[SSR-UNO] 연결되지 않음, 재시도...") # 너무 자주 출력될 수 있음
            # self.connect() # 제어 루프가 멈출 수 있으므로 주석 처리
            return "정지 (연결X)"

        try:
            # 아두이노가 쉽게 파싱할 수 있도록 명령어 뒤에 \n (줄바꿈) 추가
            cmd_str = (cmd + "\n").encode('utf-8')
            self.ser.write(cmd_str)
            self.ser.flush() # 버퍼 비우기 (즉시 전송)

            if cmd == "L": return "약풍"
            elif cmd == "M": return "중풍"
            elif cmd == "H": return "강풍"
            else: return "정지"
        
        except Exception as e:
            print(f"[SSR-UNO] 시리얼 전송 에러: {e}")
            if self.ser:
                self.ser.close()
            self.ser = None
            self.is_connected = False
            return "정지 (에러)"

    def cleanup(self):
        """종료 처리"""
        if self.is_connected and self.ser and self.ser.is_open:
            print("[SSR-UNO] 정지 명령 전송 및 연결 종료")
            self.all_off()
            time.sleep(0.1) # 마지막 명령이 전송될 시간
            self.ser.close()

# ==================================================
# 제스처 인식
# ==================================================
class GestureRecognizer:
    def __init__(self):
        self.tip_ids = [4, 8, 12, 16, 20]
        self.pip_ids = [3, 6, 10, 14, 18]
    
    def get_finger_status(self, lm_list, handedness):
        if not lm_list or len(lm_list) < 21:
            return "NONE"
        
        try:
            fingers = []
            
            # 엄지
            thumb_tip = lm_list[self.tip_ids[0]]
            thumb_ip = lm_list[self.pip_ids[0]]
            
            if handedness == "Right":
                fingers.append(1 if thumb_tip[0] < thumb_ip[0] else 0)
            else:
                fingers.append(1 if thumb_tip[0] > thumb_ip[0] else 0)
            
            # 나머지
            for i in range(1, 5):
                tip_y = lm_list[self.tip_ids[i]][1]
                mcp_y = lm_list[6 if i == 1 else self.pip_ids[i]][1]
                fingers.append(1 if tip_y < mcp_y else 0)
        
        except Exception:
            return "NONE"
        
        # 패턴 매칭
        if fingers == [1, 1, 1, 1, 1]:
            return "CHECK"
        if fingers == [1, 0, 0, 0, 0]:
            return "GOOD"
        if fingers == [0, 1, 0, 0, 1]: # (기존) 4손가락 -> 검지+새끼로 수정됨
            return "TOGGLE_LOCK"
        if fingers == [1, 0, 0, 0, 1]:
            return "TOGGLE_ROTATION"
        
        if fingers[0] == 0:
            main_fingers = fingers[1:4]
            count = sum(main_fingers)
            
            if count == 1 and main_fingers == [1, 0, 0]:
                return "FIRST_FINGER"
            elif count == 2 and main_fingers == [1, 1, 0]:
                return "SECOND_FINGER"
            elif count == 3 and main_fingers == [1, 1, 1]:
                return "THIRD_FINGER"
        
        return "NONE"

# ==================================================
# PRIZM 연결
# ==================================================
def connect_prizm():
    try:
        if PRIZM_PORT:
            try:
                ser = serial.Serial(PRIZM_PORT, BAUD, timeout=1)
                time.sleep(1)
                print(f"[PRIZM] 연결: {PRIZM_PORT}")
                return ser, PRIZM_PORT
            except:
                pass
        
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        for port in ports:
            # 아두이노 포트와 겹치지 않도록 방지
            if port == UNO_PORT:
                continue
                
            try:
                ser = serial.Serial(port, BAUD, timeout=1)
                time.sleep(1)
                print(f"[PRIZM] 연결: {port}")
                return ser, port
            except:
                continue
        
        print("[PRIZM] 없음")
        return None, None
    except Exception as e:
        print(f"[PRIZM] 에러: {e}")
        return None, None

# ==================================================
# 제어 로직
# ==================================================
def handle_thermal_mode(mlx_frame, temp_min, temp_max):
    speed_cmd = "S"
    target_temp = None
    hot_spot = None
    
    if mlx_frame is not None:
        valid_mask = (mlx_frame >= temp_min) & (mlx_frame <= temp_max)
        if np.any(valid_mask):
            valid_temps = mlx_frame[valid_mask]
            num_pixels = min(5, len(valid_temps))
            top_temps = np.sort(valid_temps)[-num_pixels:]
            target_temp = float(np.mean(top_temps))
            
            human_frame = np.where(valid_mask, mlx_frame, 0)
            h, w = mlx_frame.shape
            idx = np.argmax(human_frame)
            y, x = np.unravel_index(idx, (h, w))
            hot_spot = (x, y)
            
            if target_temp >= TEMP_THRESHOLD_H:
                speed_cmd = "H"
            elif target_temp >= TEMP_THRESHOLD_M:
                speed_cmd = "M"
            elif target_temp >= TEMP_THRESHOLD_L:
                speed_cmd = "L"
            else:
                speed_cmd = "S"
    
    return speed_cmd, target_temp, hot_spot

def handle_gesture_mode(gesture_status):
    if gesture_status == "FIRST_FINGER":
        return "L"
    if gesture_status == "SECOND_FINGER":
        return "M"
    if gesture_status == "THIRD_FINGER":
        return "H"
    return None

# ==================================================
# MLX 스레드
# ==================================================
def mlx_thread_fn():
    def init_mlx():
        try:
            _i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
            time.sleep(0.1)
            _mlx = adafruit_mlx90640.MLX90640(_i2c)
            _mlx.refresh_rate = MLX_REFRESH
            print("[MLX] 초기화 완료")
            return _i2c, _mlx
        except Exception as e:
            print(f"[MLX] 초기화 실패: {e}")
            return None, None
    
    i2c, mlx = init_mlx()
    if not mlx:
        print("[MLX] MLX 없이 계속")
        while True:
            with state_lock:
                if not state["running"]:
                    break
            time.sleep(1)
        return
    
    frame_buf = [0.0] * 768
    consecutive_errors = 0
    
    while True:
        with state_lock:
            if not state["running"]:
                break
        
        try:
            mlx.getFrame(frame_buf)
            arr = np.array(frame_buf, dtype=np.float32).reshape((24, 32))
            
            with state_lock:
                state["mlx_frame"] = arr
            
            consecutive_errors = 0
        
        except OSError as e:
            consecutive_errors += 1
            print(f"[MLX] I2C 에러 ({consecutive_errors}): {e}")
            
            if consecutive_errors >= 10:
                print("[MLX] 재초기화...")
                try:
                    if i2c:
                        i2c.deinit()
                except:
                    pass
                time.sleep(1)
                i2c, mlx = init_mlx()
                consecutive_errors = 0
            
            time.sleep(0.5)
        
        except Exception as e:
            print(f"[MLX] 예외: {e}")
            time.sleep(1)

# ==================================================
# Pose 스레드
# ==================================================
def pose_thread_fn():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    recognizer = GestureRecognizer()
    
    cap = cv2.VideoCapture(RGB_CAM_INDEX)
    if not cap.isOpened():
        print(f"[POSE] 카메라 실패 (인덱스 {RGB_CAM_INDEX})")
        with state_lock:
            state["running"] = False
        return
    
    print("[POSE] 손 인식 시작")
    
    last_gesture = "NONE"
    last_gesture_time = 0.0
    
    while True:
        with state_lock:
            if not state["running"]:
                break
        
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        gesture = "NONE"
        
        if results.multi_hand_landmarks:
            landmarks_obj = results.multi_hand_landmarks[0]
            lm_list = [(lm.x, lm.y, lm.z) for lm in landmarks_obj.landmark]
            hand_label = results.multi_handedness[0].classification[0].label
            
            gesture = recognizer.get_finger_status(lm_list, hand_label)
            
            now = time.time()
            if gesture != last_gesture:
                last_gesture = gesture
                last_gesture_time = now
            else:
                if (now - last_gesture_time) >= GESTURE_STABLE_TIME:
                    with state_lock:
                        if state["gesture_status"] != gesture:
                            state["last_detected_gesture_time"] = now
                        state["gesture_status"] = gesture
                        state["detected_hand_landmarks"] = landmarks_obj
        else:
            gesture = "NONE"
            last_gesture = "NONE"
            with state_lock:
                state["gesture_status"] = "NONE"
                state["detected_hand_landmarks"] = None
        
        time.sleep(0.03)
    
    if cap:
        cap.release()
    print("[POSE] 종료")

# ==================================================
# Control 스레드
# ==================================================
def control_thread_fn(ssr):
    ser_prizm, prizm_port = connect_prizm()
    
    with state_lock:
        state["prizm_connected"] = (ser_prizm is not None)
    
    print("[CONTROL] 시작")
    
    while True:
        try:
            with state_lock:
                if not state["running"]:
                    break
                s = state.copy()
            
            current_ts = time.time()
            gesture = s["gesture_status"]
            
            # 회전 토글 (3초 + GOOD)
            with state_lock:
                if state["gesture_status"] == "TOGGLE_ROTATION":
                    if not state["rotation_selecting"]:
                        state["rotation_selecting"] = True
                        state["rotation_select_start"] = current_ts
                        print("[ROTATION] 회전 선택 시작 (3초)")
                
                elif state["rotation_selecting"] and state["gesture_status"] not in ["TOGGLE_ROTATION", "GOOD"]:
                    state["rotation_selecting"] = False
                    state["selected_rotation_action"] = None
                    print("[ROTATION] 선택 취소")
                
                if state["rotation_selecting"] and not state["selected_rotation_action"]:
                    if (current_ts - state["rotation_select_start"]) >= 3.0:
                        state["selected_rotation_action"] = "stop" if state["is_rotating"] else "start"
                        action = "정지" if state["selected_rotation_action"] == "stop" else "시작"
                        print(f"[ROTATION] {action} 선택 (GOOD으로 확정)")
                
                if state["selected_rotation_action"] and state["gesture_status"] == "GOOD":
                    if state["selected_rotation_action"] == "start":
                        state["is_rotating"] = True
                        print("[ROTATION] ★ 회전 시작 ★")
                    else:
                        state["is_rotating"] = False
                        print("[ROTATION] ★ 회전 정지 ★")
                    
                    state["rotation_selecting"] = False
                    state["selected_rotation_action"] = None
                    state["last_toggle_rotation_ts"] = current_ts
                    time.sleep(0.2)
            
            # 잠금 토글 (3초 + GOOD)
            with state_lock:
                if state["gesture_status"] == "TOGGLE_LOCK":
                    if not state["lock_selecting"]:
                        state["lock_selecting"] = True
                        state["lock_select_start"] = current_ts
                        print("[LOCK] 잠금 선택 시작 (3초)")
                
                elif state["lock_selecting"] and state["gesture_status"] not in ["TOGGLE_LOCK", "GOOD"]:
                    state["lock_selecting"] = False
                    state["selected_lock_action"] = None
                    print("[LOCK] 선택 취소")
                
                if state["lock_selecting"] and not state["selected_lock_action"]:
                    if (current_ts - state["lock_select_start"]) >= 3.0:
                        state["selected_lock_action"] = "unlock" if state["is_locked"] else "lock"
                        action = "해제" if state["selected_lock_action"] == "unlock" else "잠금"
                        print(f"[LOCK] {action} 선택 (GOOD으로 확정)")
                
                if state["selected_lock_action"] and state["gesture_status"] == "GOOD":
                    if state["selected_lock_action"] == "lock":
                        state["locked_speed"] = state["last_speed_cmd"]
                        state["is_locked"] = True
                        print(f"[LOCK] ★ 잠금 완료 ({state['locked_speed']}) ★")
                    else:
                        state["is_locked"] = False
                        print("[LOCK] ★ 잠금 해제 ★")
                    
                    state["lock_selecting"] = False
                    state["selected_lock_action"] = None
                    state["last_toggle_lock_ts"] = current_ts
                    time.sleep(0.2)
            
            # 모드 전환 (CHECK → GOOD)
            if not s["is_locked"]:
                with state_lock:
                    if state["gesture_status"] == "CHECK":
                        if not state["mode_selecting"]:
                            state["mode_selecting"] = True
                            state["mode_select_start"] = current_ts
                            print("[MODE] 모드 선택 시작")
                    
                    elif state["mode_selecting"] and state["gesture_status"] not in ["CHECK", "GOOD"]:
                        state["mode_selecting"] = False
                        state["selected_mode"] = None
                        print("[MODE] 선택 취소")
                    
                    if state["mode_selecting"] and not state["selected_mode"]:
                        if (current_ts - state["mode_select_start"]) >= 3.0:
                            mode_order = ["THERMAL", "GESTURE"]
                            idx = mode_order.index(state["control_mode"])
                            next_idx = (idx + 1) % len(mode_order)
                            state["selected_mode"] = mode_order[next_idx]
                            print(f"[MODE] {state['selected_mode']} 선택")
                    
                    if state["selected_mode"] and state["gesture_status"] == "GOOD":
                        state["control_mode"] = state["selected_mode"]
                        state["mode_selecting"] = False
                        state["selected_mode"] = None
                        print(f"[MODE] ★ {state['control_mode']} ★")
            
            # 회전 명령 (PRIZM)
            turn_cmd = "T" if s["is_rotating"] else "S"
            if turn_cmd != s["last_turn_cmd"]:
                try:
                    if ser_prizm and ser_prizm.is_open:
                        ser_prizm.write((turn_cmd + "\n").encode())
                        ser_prizm.flush()
                        with state_lock:
                            state["last_turn_cmd"] = turn_cmd
                        print(f"[TURN] {turn_cmd}")
                except Exception as e:
                    print(f"[PRIZM] 에러: {e}")
            
            # 풍량 제어
            speed_cmd = s["last_speed_cmd"]
            
            if s["is_locked"]:
                speed_cmd = s["locked_speed"]
            else:
                temp = None
                spot = None
                
                if s["control_mode"] == "THERMAL":
                    speed_cmd, temp, spot = handle_thermal_mode(
                        s["mlx_frame"],
                        HUMAN_TEMP_MIN,
                        HUMAN_TEMP_MAX
                    )
                    if speed_cmd != s["last_speed_cmd"]:
                        temp_str = f"{temp:.1f}" if temp else "N/A"
                        print(f"[THERMAL] {temp_str}°C → {speed_cmd}")
                
                elif s["control_mode"] == "GESTURE":
                    new_speed = handle_gesture_mode(s["gesture_status"])
                    if new_speed is not None:
                        speed_cmd = new_speed
                        if speed_cmd != s["last_speed_cmd"]:
                            print(f"[GESTURE] {s['gesture_status']} → {speed_cmd}")
                
                with state_lock:
                    state["target_temp"] = temp
                    state["hot_spot_coords"] = spot
            
            # SSR 제어 (변경시에만)
            if speed_cmd != s["last_speed_cmd"]:
                try:
                    name = ssr.set_speed(speed_cmd) # 이제 아두이노로 명령 전송
                    with state_lock:
                        state["last_speed_cmd"] = speed_cmd
                    print(f"[SSR-UNO] {speed_cmd} ({name})")
                except Exception as e:
                    print(f"[SSR-UNO] 에러: {e}")
            
            time.sleep(PD_TS)
        
        except Exception as e:
            print(f"[CONTROL] 에러: {e}")
            time.sleep(1)
    
    # 종료
    print("[CONTROL] 종료")
    if ser_prizm:
        try:
            ser_prizm.write(b"S\n")
            ser_prizm.close()
        except:
            pass

# ==================================================
# 시각화 스레드
# ==================================================
def vis_thread_fn():
    w_up, h_up = 640, 480
    COLORMAP = cv2.COLORMAP_JET
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    print("[VIS] 시작")
    
    while True:
        with state_lock:
            local_state = state.copy()
            if not local_state["running"]:
                break
        
        if local_state["mlx_frame"] is not None:
            frame = local_state["mlx_frame"]
            norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            thermal_up = cv2.resize(
                cv2.applyColorMap(norm, COLORMAP),
                (w_up, h_up),
                interpolation=cv2.INTER_CUBIC
            )
            
            if local_state["detected_hand_landmarks"]:
                try:
                    mp_drawing.draw_landmarks(
                        thermal_up,
                        local_state["detected_hand_landmarks"],
                        mp_hands.HAND_CONNECTIONS
                    )
                except:
                    pass
            
            if local_state["hot_spot_coords"]:
                h_s, w_s = h_up / 24, w_up / 32
                x, y = local_state["hot_spot_coords"]
                sx, sy = int(x * w_s), int(y * h_s)
                size = int(3 * w_s)
                cv2.rectangle(
                    thermal_up,
                    (sx - size // 2, sy - size // 2),
                    (sx + size // 2, sy + size // 2),
                    (255, 255, 255),
                    2
                )
            
            if local_state["target_temp"]:
                cv2.putText(
                    thermal_up,
                    f"Temp: {local_state['target_temp']:.1f}C",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )
            
            rotation = "ROTATING" if local_state['is_rotating'] else "STOPPED"
            lock = "LOCKED" if local_state['is_locked'] else "UNLOCKED"
            prizm = "✓" if local_state.get('prizm_connected', False) else "✗"
            
            info1 = f"MODE: {local_state['control_mode']} | GESTURE: {local_state['gesture_status']}"
            info2 = f"SPEED: {local_state['last_speed_cmd']} | ROT: {rotation} | LOCK: {lock}"
            info3 = f"PRIZM: {prizm}"

            cv2.putText(thermal_up, info1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(thermal_up, info2, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(thermal_up, info3, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            
            if local_state['mode_selecting']:
                if local_state['selected_mode']:
                    text = f"Change to {local_state['selected_mode']}?"
                else:
                    text = "Keep CHECK for 3s..."
                cv2.putText(thermal_up, text, (150, h_up // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            
            if local_state.get('lock_selecting', False):
                if local_state.get('selected_lock_action'):
                    action = "Unlock?" if local_state['selected_lock_action'] == "unlock" else "Lock?"
                    text = f"{action} Press GOOD"
                else:
                    text = "Keep Index+Pinky for 3s..."
                cv2.putText(thermal_up, text, (120, h_up // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 3)
            
            if local_state.get('rotation_selecting', False):
                if local_state.get('selected_rotation_action'):
                    action = "Stop?" if local_state['selected_rotation_action'] == "stop" else "Start?"
                    text = f"{action} Press GOOD"
                else:
                    text = "Keep Thumb+Pinky for 3s..."
                cv2.putText(thermal_up, text, (100, h_up // 2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 255, 0), 3)
            
            cv2.imshow("THERMAL", thermal_up)
        
        if cv2.waitKey(1) & 0xFF == 27:
            with state_lock:
                state["running"] = False
            break
        
        time.sleep(VIS_INTERVAL)
    
    cv2.destroyAllWindows()
    print("[VIS] 종료")

# ==================================================
# 메인
# ==================================================
def main():
    print("=" * 60)
    print("★ 아두이노(SSR) 연동 선풍기 시스템 ★")
    print("=" * 60)
    print(f"\n제어 포트:")
    print(f"  Arduino: {UNO_PORT}")
    print(f"  PRIZM: {PRIZM_PORT}")
    print("\n제스처:")
    print("  검지 1개: 약풍")
    print("  검지+중지: 중풍")
    print("  검지+중지+약지: 강풍")
    print("  엄지+새끼 3초: 회전")
    print("  검지+새끼 3초: 잠금")
    print("\nESC: 종료")
    print("=" * 60 + "\n")
    
    # 음성 제어 모듈 임포트
    from voice_control import voice_thread_fn  # ✅ thread 철자 확인!
    
    # SSR 초기화
    try:
        ssr = SSRController(port=UNO_PORT, baud=BAUD)
        if not ssr.is_connected:
             print("❌ SSR(Uno)가 연결되지 않았지만, 다른 스레드는 시작합니다.")
    except Exception as e:
        print(f"❌ SSR(Uno) 초기화 중 심각한 오류: {e}")
        return
    
    # 스레드 시작
    threads = [
        threading.Thread(target=mlx_thread_fn, name="MLX"),
        threading.Thread(target=pose_thread_fn, name="Pose"),
        threading.Thread(target=control_thread_fn, args=(ssr,), name="Control"),  # ✅ 쉼표 추가!
        threading.Thread(target=voice_thread_fn, args=(ssr, state, state_lock), name="Voice")  # ✅ 대문자 V
    ]
    
    for t in threads:
        t.start()
        print(f"[MAIN] {t.name} 시작")
    
    try:
        vis_thread_fn()
    except KeyboardInterrupt:
        with state_lock:
            state["running"] = False
        print("\n\n[MAIN] 사용자 중단")
    
    print("[MAIN] 종료 대기...")
    for t in threads:
        t.join(timeout=5)
    
    # SSR 종료
    ssr.cleanup()
    print("[MAIN] SSR(Uno) 종료 완료")
    
    print("\n" + "=" * 60)
    print("★ 종료 완료 ★")
    print("=" * 60)

if __name__ == "__main__":
    main()
