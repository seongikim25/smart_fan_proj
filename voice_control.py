"""
HumaWind - Voice Control Module
음성 대화 인터페이스를 기존 시스템에 병렬 스레드로 추가

사용법:
1. pip install openai speechrecognition gtts pyaudio --break-system-packages
2. sudo apt-get install mpg123
3. export OPENAI_API_KEY="your-api-key-here"
4. total_code.py의 main()에서 voice_thread 추가
"""

import os
import time
import json
import threading
import tempfile
import speech_recognition as sr
from openai import OpenAI
from gtts import gTTS
import subprocess

# ==================================================
# 설정
# ==================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
LANGUAGE = "ko"  # 한국어
LISTEN_TIMEOUT = 5  # 음성 입력 대기 시간 (초)
PHRASE_TIME_LIMIT = 10  # 최대 녹음 시간 (초)

# 웨이크 워드 설정
WAKE_WORDS = ["휴마윈드", "풍기야", "선풍기"]  # 이 중 하나를 말하면 활성화
WAKE_TIMEOUT = 10  # 웨이크 워드 활성화 지속 시간 (초)

# ==================================================
# LLM 프롬프트
# ==================================================
SYSTEM_PROMPT = """당신은 HumaWind 스마트 선풍기의 음성 비서입니다.
사용자의 한국어 요청을 이해하고 자연스럽게 응답하며, 필요시 선풍기를 제어합니다.

**제어 명령어:**
- L: 약풍 (미풍)
- M: 중풍
- H: 강풍
- S: 정지 (끄기)
- R: 회전 토글 (회전 ON/OFF)

**응답 규칙:**
1. 사용자 의도를 파악하여 적절한 명령을 매핑합니다.
2. 선풍기 제어가 필요 없는 대화(날씨, 잡담 등)는 command를 null로 설정합니다.
3. 응답은 짧고 자연스럽게 (1-2문장).
4. 항상 JSON 형식으로만 출력합니다.

**출력 형식 (JSON only):**
{
  "reply": "사용자에게 할 음성 응답 (한국어)",
  "command": "L|M|H|S|R|null"
}

**예시:**
사용자: "나 더워"
→ {"reply": "선풍기를 미풍으로 켭니다.", "command": "L"}

사용자: "좀 더 세게"
→ {"reply": "중풍으로 조정합니다.", "command": "M"}

사용자: "강풍으로 틀어줘"
→ {"reply": "강풍으로 변경합니다.", "command": "H"}

사용자: "꺼줘"
→ {"reply": "선풍기를 끕니다.", "command": "S"}

사용자: "회전해줘" 또는 "돌려줘" 또는 "회전 시작"
→ {"reply": "회전 기능을 켭니다.", "command": "R"}

사용자: "회전 멈춰" 또는 "회전 정지"
→ {"reply": "회전을 멈춥니다.", "command": "R"}

사용자: "오늘 날씨 어때?"
→ {"reply": "죄송하지만 날씨 정보는 제공할 수 없습니다. 선풍기 제어만 도와드릴 수 있어요.", "command": null}

사용자: "안녕"
→ {"reply": "안녕하세요! 선풍기 조절이 필요하신가요?", "command": null}

**중요:** 반드시 유효한 JSON만 출력하고, 추가 설명이나 마크다운을 포함하지 마세요."""

# ==================================================
# 음성 제어 클래스
# ==================================================
class VoiceController:
    def __init__(self, ssr_controller, state_dict, state_lock):
        """
        ssr_controller: SSRController 인스턴스
        state_dict: 공용 state 딕셔너리
        state_lock: threading.Lock 객체
        """
        self.ssr = ssr_controller
        self.state = state_dict
        self.lock = state_lock
        
        # OpenAI 클라이언트 초기화
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # 음성 인식기 초기화
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000  # 주변 소음에 따라 조정
        self.recognizer.dynamic_energy_threshold = True
        
        # 웨이크 워드 활성화 상태 추적
        self.wake_active_until = 0  # 웨이크 워드 활성화 만료 시간
        
        print("[VOICE] 초기화 완료")
    
    def check_wake_word(self, text):
        """웨이크 워드 감지 또는 활성 상태 확인"""
        current_time = time.time()
        
        # 이미 활성화 상태인지 확인
        if current_time < self.wake_active_until:
            remaining = int(self.wake_active_until - current_time)
            print(f"[VOICE] ⏰ 활성 상태 유지 중 (남은 시간: {remaining}초)")
            return True
        
        # 웨이크 워드 체크
        if not text:
            return False
        
        text_lower = text.lower()
        for wake_word in WAKE_WORDS:
            if wake_word in text_lower:
                # 웨이크 워드 감지 → 타임아웃 설정
                self.wake_active_until = current_time + WAKE_TIMEOUT
                print(f"[VOICE] ✅ 웨이크 워드 감지: '{wake_word}' → {WAKE_TIMEOUT}초 동안 활성화")
                return True
        
        return False
    
    def listen_and_transcribe(self):
        """마이크로 음성 입력 받아 텍스트로 변환"""
        with sr.Microphone() as source:
            print("[VOICE] 🎤 듣는 중...")
            
            # 주변 소음 조정
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                audio = self.recognizer.listen(
                    source, 
                    timeout=LISTEN_TIMEOUT,
                    phrase_time_limit=PHRASE_TIME_LIMIT
                )
                
                # Google Web Speech API로 변환
                text = self.recognizer.recognize_google(audio, language=LANGUAGE)
                print(f"[VOICE] 인식됨: {text}")
                return text
            
            except sr.WaitTimeoutError:
                return None  # 타임아웃 (조용히 무시)
            
            except sr.UnknownValueError:
                print("[VOICE] ❌ 음성을 인식하지 못했습니다.")
                return None
            
            except sr.RequestError as e:
                print(f"[VOICE] ❌ Google API 오류: {e}")
                return None
    
    def get_llm_response(self, user_text):
        """LLM에게 사용자 입력 전달 → JSON 응답 받기"""
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            raw_content = response.choices[0].message.content.strip()
            print(f"[VOICE] LLM 응답: {raw_content}")
            
            # JSON 파싱
            # 혹시 마크다운 코드블록이 있으면 제거
            if raw_content.startswith("```"):
                lines = raw_content.split("\n")
                raw_content = "\n".join(lines[1:-1])  # 첫/마지막 줄 제거
            
            data = json.loads(raw_content)
            
            reply = data.get("reply", "")
            command = data.get("command")
            
            # command가 문자열 "null"인 경우 처리
            if command == "null" or command is None:
                command = None
            
            return reply, command
        
        except json.JSONDecodeError as e:
            print(f"[VOICE] ❌ JSON 파싱 실패: {e}")
            return "죄송해요, 응답을 처리하지 못했습니다.", None
        
        except Exception as e:
            print(f"[VOICE] ❌ LLM 오류: {e}")
            return "죄송해요, 오류가 발생했습니다.", None
    
    def speak(self, text):
        """TTS로 음성 출력 (gTTS + mpg123)"""
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                temp_file = fp.name
            
            # gTTS로 음성 합성
            tts = gTTS(text=text, lang=LANGUAGE, slow=False)
            tts.save(temp_file)
            
            # mpg123로 재생
            print(f"[VOICE] 🔊 말하는 중: {text}")
            subprocess.run(["mpg123", "-q", temp_file], check=True)
            
            # 임시 파일 삭제
            os.remove(temp_file)
        
        except Exception as e:
            print(f"[VOICE] ❌ TTS 오류: {e}")
    
    def execute_command(self, command):
        """선풍기 제어 명령 실행"""
        if command is None:
            return
        
        command = command.upper()
        
        # 회전 명령 처리
        if command == "R":
            with self.lock:
                current_rotating = self.state.get("is_rotating", False)
                self.state["is_rotating"] = not current_rotating
                new_state = "시작" if not current_rotating else "정지"
            print(f"[VOICE] ✅ 회전 {new_state}")
            return
        
        # 풍량 명령 처리
        if command not in ["L", "M", "H", "S"]:
            print(f"[VOICE] ❌ 잘못된 명령: {command}")
            return
        
        try:
            # SSR 제어
            speed_name = self.ssr.set_speed(command)
            
            # state 업데이트 (기존 시스템과 동기화)
            with self.lock:
                self.state["last_speed_cmd"] = command
            
            print(f"[VOICE] ✅ 명령 실행: {command} ({speed_name})")
        
        except Exception as e:
            print(f"[VOICE] ❌ 명령 실행 오류: {e}")
    
    def run_loop(self):
        """메인 루프 (스레드에서 실행)"""
        print("[VOICE] 🎙️ 음성 제어 시작")
        print(f"[VOICE] 💡 웨이크 워드: {', '.join(WAKE_WORDS)}")
        print(f"[VOICE] ⏰ 활성화 시간: {WAKE_TIMEOUT}초")
        print("[VOICE] 사용법:")
        print("  1. '풍기야' → 10초 동안 활성화")
        print("  2. '나 더워', '강풍', '회전' 등 연속 명령 가능")
        print("  또는 '풍기야 나 더워' 한 번에 말해도 OK!")
        
        while True:
            # 시스템 종료 체크
            with self.lock:
                if not self.state["running"]:
                    break
            
            # 1. 음성 입력
            user_text = self.listen_and_transcribe()
            if not user_text:
                continue
            
            # 2. 웨이크 워드 체크 (활성화 또는 타임아웃 연장)
            if not self.check_wake_word(user_text):
                print(f"[VOICE] ⏸️ 비활성 상태, 무시: '{user_text}'")
                continue
            
            # 3. 웨이크 워드만 말한 경우 (명령 없음)
            text_lower = user_text.lower()
            is_wake_only = any(wake_word in text_lower and len(user_text.strip()) <= len(wake_word) + 2 
                              for wake_word in WAKE_WORDS)
            
            if is_wake_only:
                print(f"[VOICE] 🎯 대기 중... 명령을 말씀해주세요")
                # TTS로 응답
                import random
                responses = ["네?", "부르셨나요?", "네, 듣고 있어요"]
                self.speak(random.choice(responses))
                continue
            
            print(f"[VOICE] 🎯 명령 처리 중...")
            
            # 4. LLM 처리
            reply, command = self.get_llm_response(user_text)
            
            # 5. 음성 응답
            if reply:
                self.speak(reply)
            
            # 6. 명령 실행
            self.execute_command(command)
            
            time.sleep(0.5)  # 짧은 대기
        
        print("[VOICE] 종료")

# ==================================================
# 스레드 함수 (total_code.py에서 호출)
# ==================================================
def voice_thread_fn(ssr_controller, state_dict, state_lock):
    """
    기존 시스템에 추가할 음성 제어 스레드
    
    Parameters:
        ssr_controller: SSRController 인스턴스
        state_dict: 공용 state 딕셔너리
        state_lock: threading.Lock 객체
    """
    try:
        controller = VoiceController(ssr_controller, state_dict, state_lock)
        controller.run_loop()
    except KeyboardInterrupt:
        print("\n[VOICE] 사용자 중단")
    except Exception as e:
        print(f"[VOICE] ❌ 치명적 오류: {e}")
        with state_lock:
            state_dict["running"] = False

# ==================================================
# 독립 실행 테스트 (선택사항)
# ==================================================
if __name__ == "__main__":
    print("=" * 60)
    print("★ HumaWind 음성 제어 테스트 ★")
    print("=" * 60)
    
    # 더미 SSR 컨트롤러 (테스트용)
    class DummySSR:
        def set_speed(self, cmd):
            speeds = {"L": "약풍", "M": "중풍", "H": "강풍", "S": "정지"}
            return speeds.get(cmd, "알 수 없음")
    
    # 더미 state
    test_state = {
        "running": True,
        "last_speed_cmd": "S"
    }
    test_lock = threading.Lock()
    
    # 음성 제어 시작
    voice_thread_fn(DummySSR(), test_state, test_lock)
