# -*- coding: utf-8 -*-
"""
==========================================================================
 TRO CHOI BUA - KEO - BAO (ROCK - PAPER - SCISSORS) QUA CAMERA
 Cong nghe: OpenCV (cv2) + MediaPipe Hands (chi lay 21 landmark)
 Thuat toan nhan dien cu chi: TU CODE bang phep so sanh toa do (khong dung
 Gesture Recognizer, khong dung tich vo huong / goc vector / luong giac).
==========================================================================
Cai dat:   pip install opencv-python mediapipe
Chay:      python main.py
Phim:      SPACE = bat dau hiep | R = choi lai | Q hoac ESC = thoat

Luu y: cv2.putText KHONG ho tro chu co dau tieng Viet, nen toan bo chu
hien thi tren man hinh duoc viet KHONG DAU. Phan comment trong code thi
viet co dau de ban doc cho de hieu.
"""

import time
import random
import math
from collections import deque, Counter

import cv2
import mediapipe as mp

# ==========================================================================
# 1. HANG SO CAU HINH
# ==========================================================================
WINNING_SCORE = 3          # Ai dat 3 diem truoc la thang chung cuoc
COUNTDOWN_SECONDS = 3      # Dem nguoc 3 giay
RESULT_SECONDS = 2.5       # Hien ket qua hiep trong 2.5 giay
HISTORY_LEN = 5            # So frame gan nhat dung de "lam muot" cu chi

# Cac trang thai cua tro choi (State Machine)
WAITING = "WAITING"
COUNTDOWN = "COUNTDOWN"
RESULT = "RESULT"
GAME_OVER = "GAME_OVER"

# Mau (OpenCV dung he BGR: Blue - Green - Red)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (80, 220, 80)
RED = (60, 60, 240)
YELLOW = (0, 220, 255)
ORANGE = (0, 140, 255)
GRAY = (60, 60, 60)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ==========================================================================
# 2. THUAT TOAN NHAN DIEN CU CHI (PHAN TU CODE)
# ==========================================================================
def get_finger_states(hand_landmarks, handedness_label=None):
    """
    Tra ve danh sach 5 phan tu [Cai, Tro, Giua, Ap_ut, Ut]
    voi 1 = ngon DUOI (mo), 0 = ngon GAP (nam).

    Nhac lai he toa do anh trong OpenCV / MediaPipe:
      - Goc (0,0) o GOC TREN BEN TRAI.
      - x tang tu TRAI sang PHAI, y tang tu TREN xuong DUOI.
      => Diem nam CAO hon tren man hinh thi co gia tri y NHO hon.

    MediaPipe tra toa do da CHUAN HOA (0.0 -> 1.0), ta chi so sanh hai diem
    voi nhau nen khong can doi sang pixel.

    Tham so handedness_label ("Left"/"Right") duoc giu lai cho dung chu ky
    ham de de mo rong. Thuat toan duoi day KHONG phu thuoc vao no, nen dung
    cho ca tay trai, tay phai, co lat guong (cv2.flip) hay khong.
    """
    lm = hand_landmarks.landmark  # lm[i].x, lm[i].y la toa do diem thu i
    states = []

    # ---------- NGON CAI (di chuyen NGANG nen so sanh theo truc X) ----------
    # Y tuong: khi DUOI, dau ngon cai (diem 4) boc ra XA phia ngon ut hon so
    # voi khop IP (diem 3). Khi GAP, ngon cai thu vao long ban tay nen dau
    # ngon (4) o GAN ngon ut (diem 17) hon khop IP (3).
    # Dung abs() (gia tri tuyet doi) nen ket qua dung cho ca hai tay.
    khoang_cach_dau_ngon = abs(lm[4].x - lm[17].x)
    khoang_cach_khop_ip = abs(lm[3].x - lm[17].x)
    states.append(1 if khoang_cach_dau_ngon > khoang_cach_khop_ip else 0)

    # ---------- 4 NGON DAI: Tro, Giua, Ap ut, Ut (so sanh theo truc Y) ------
    # Cap (dau ngon, khop PIP): Tro (8,6) Giua (12,10) Ap ut (16,14) Ut (20,18)
    # Ban tay gio THANG DUNG:
    #   - Ngon duoi: dau ngon vuon CAO hon khop PIP  => y_dau < y_PIP => 1
    #   - Ngon gap : dau ngon cuon XUONG THAP hon PIP => y_dau > y_PIP => 0
    cac_cap_ngon = [(8, 6), (12, 10), (16, 14), (20, 18)]
    for tip_id, pip_id in cac_cap_ngon:
        states.append(1 if lm[tip_id].y < lm[pip_id].y else 0)

    return states


def classify_gesture(finger_states):
    """
    Anh xa mang 5 ngon tay -> "Rock" / "Paper" / "Scissors" / "Unknown".

    Ngon cai (finger_states[0]) bi BO QUA vi no nho, di chuyen ngang, rat de
    nhieu. Moi cu chi da phan biet duoc chi bang 4 ngon dai.

        Rock     : [*,0,0,0,0]  4 ngon dai deu gap
        Paper    : [*,1,1,1,1]  4 ngon dai deu duoi
        Scissors : [*,1,1,0,0]  chi tro va giua duoi
        Unknown  : cac to hop con lai
    """
    tro, giua, ap_ut, ut = finger_states[1], finger_states[2], \
        finger_states[3], finger_states[4]

    if tro == 0 and giua == 0 and ap_ut == 0 and ut == 0:
        return "Rock"
    if tro == 1 and giua == 1 and ap_ut == 1 and ut == 1:
        return "Paper"
    if tro == 1 and giua == 1 and ap_ut == 0 and ut == 0:
        return "Scissors"
    return "Unknown"


def determine_winner(player_move, computer_move):
    """
    So sanh theo luat Bua - Keo - Bao. Tra ve:
      "draw"     : hoa
      "player"   : nguoi choi thang
      "computer" : may tinh thang

    Luat: Rock thang Scissors, Scissors thang Paper, Paper thang Rock.
    """
    if player_move == computer_move:
        return "draw"

    # Tap cac cap (nguoi, may) ma nguoi choi THANG
    cac_truong_hop_thang = {
        ("Rock", "Scissors"),
        ("Scissors", "Paper"),
        ("Paper", "Rock"),
    }
    if (player_move, computer_move) in cac_truong_hop_thang:
        return "player"
    return "computer"


# ==========================================================================
# 3. CAC HAM VE GIAO DIEN (HUD)
# ==========================================================================
def put_text_center(img, text, center_x, center_y, scale, color, thickness):
    """Ve chu CAN GIUA tai (center_x, center_y), co vien den cho de doc."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    org = (int(center_x - w / 2), int(center_y + h / 2))
    cv2.putText(img, text, org, font, scale, BLACK, thickness + 4, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_move_icon(img, move, cx, cy, size, color):
    """
    Ve bieu tuong cu chi bang hinh khoi OpenCV (khong can file anh):
      Rock     -> hinh tron dac (nam dam)
      Paper    -> hinh chu nhat (to giay)
      Scissors -> hai duong cat nhau tao chu X (keo)
    """
    if move == "Rock":
        cv2.circle(img, (cx, cy), size, color, -1)
        cv2.circle(img, (cx, cy), size, WHITE, 3)
    elif move == "Paper":
        cv2.rectangle(img, (cx - size, cy - size), (cx + size, cy + size),
                      color, -1)
        cv2.rectangle(img, (cx - size, cy - size), (cx + size, cy + size),
                      WHITE, 3)
    elif move == "Scissors":
        cv2.line(img, (cx - size, cy - size), (cx + size, cy + size), color, 12)
        cv2.line(img, (cx - size, cy + size), (cx + size, cy - size), color, 12)
        cv2.circle(img, (cx - size, cy - size), 12, WHITE, 3)
        cv2.circle(img, (cx - size, cy + size), 12, WHITE, 3)
    else:
        put_text_center(img, "?", cx, cy, 2.5, color, 5)


def draw_ui(frame, state, player_score, computer_score, round_number,
            live_gesture, finger_states, locked_player_move, computer_move,
            countdown_value, message, message_color, game_over_text):
    """
    Ve toan bo giao dien len khung hinh camera:
      - Header: ti so, muc tieu, hiep dau
      - Cu chi hien tai + mang 5 ngon tay (de demo thuat toan)
      - Dem nguoc / ket qua hiep / man hinh chung cuoc
      - Lua chon cua may tinh (bieu tuong hinh khoi)
    """
    h, w = frame.shape[:2]

    # ---------- Header / Banner phia tren (nen ban trong suot) ----------
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), GRAY, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "Player: %d  -  Computer: %d"
                % (player_score, computer_score), (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "First to %d wins" % WINNING_SCORE, (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)
    round_text = "Hiep: %d" % round_number
    (tw, _), _ = cv2.getTextSize(round_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, round_text, (w - tw - 15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)

    # ---------- Cu chi hien tai cua nguoi choi + mang 5 ngon (goc duoi trai) ----------
    cv2.rectangle(frame, (0, h - 90), (330, h), GRAY, -1)
    cv2.putText(frame, "Cu chi: %s" % live_gesture, (12, h - 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2, cv2.LINE_AA)
    if finger_states is not None:
        mang = "[" + ",".join(str(s) for s in finger_states) + "]"
    else:
        mang = "[ khong thay tay ]"
    cv2.putText(frame, "Ngon: %s" % mang, (12, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "[Cai,Tro,Giua,ApUt,Ut]", (190, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1, cv2.LINE_AA)

    # ---------- Noi dung theo tung trang thai ----------
    if state == WAITING:
        put_text_center(frame, "Nhan SPACE de bat dau hiep moi | "
                        "R: Choi lai | Q: Thoat", w // 2, h - 120,
                        0.7, YELLOW, 2)

    elif state == COUNTDOWN:
        # So 3 -> 2 -> 1 to, ro o giua man hinh
        put_text_center(frame, str(countdown_value), w // 2, h // 2,
                        8, ORANGE, 15)

    elif state == RESULT:
        # Khung hien cu chi nguoi choi (trai) va may tinh (phai)
        if locked_player_move in ("Rock", "Paper", "Scissors"):
            put_text_center(frame, "BAN", w // 4, 120, 0.9, WHITE, 2)
            draw_move_icon(frame, locked_player_move, w // 4, 200, 50, GREEN)
            put_text_center(frame, locked_player_move, w // 4, 285,
                            0.9, GREEN, 2)

            put_text_center(frame, "MAY TINH", 3 * w // 4, 120, 0.9, WHITE, 2)
            draw_move_icon(frame, computer_move, 3 * w // 4, 200, 50, RED)
            put_text_center(frame, computer_move, 3 * w // 4, 285,
                            0.9, RED, 2)
        put_text_center(frame, message, w // 2, h // 2 + 40, 1.1,
                        message_color, 3)

    elif state == GAME_OVER:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), BLACK, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        put_text_center(frame, game_over_text, w // 2, h // 2 - 40,
                        1.0, YELLOW, 3)
        put_text_center(frame, "Ty so cuoi: %d - %d"
                        % (player_score, computer_score), w // 2,
                        h // 2 + 20, 0.9, WHITE, 2)
        put_text_center(frame, "Nhan R de choi van moi | Q / ESC de thoat",
                        w // 2, h // 2 + 80, 0.7, GREEN, 2)

    # Khi dang WAITING / COUNTDOWN van cho thay lua chon cu cua may (neu co)
    # -> Khong can, de giao dien gon gang.


# ==========================================================================
# 4. VONG LAP CHINH
# ==========================================================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Khong mo duoc camera! Hay kiem tra lai thiet bi.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    # ---------- Bien trang thai tro choi ----------
    state = WAITING
    player_score = 0
    computer_score = 0
    round_number = 1

    countdown_start = 0.0      # Thoi diem bat dau dem nguoc
    result_start = 0.0         # Thoi diem bat dau hien ket qua
    locked_player_move = "Unknown"
    computer_move = "Rock"
    message = ""
    message_color = WHITE
    game_over_text = ""

    # Luu N cu chi gan nhat -> lay cu chi xuat hien nhieu nhat (giam nhieu)
    gesture_history = deque(maxlen=HISTORY_LEN)

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Khong doc duoc khung hinh tu camera.")
                break

            # Lat guong de nguoi choi thao tac tu nhien nhu soi guong
            frame = cv2.flip(frame, 1)

            # MediaPipe yeu cau anh RGB (OpenCV mac dinh la BGR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # ---------- Lay cu chi hien tai tu landmark ----------
            finger_states = None
            raw_gesture = "Unknown"
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                label = None
                if results.multi_handedness:
                    label = results.multi_handedness[0].classification[0].label

                # Ve cac khop va duong noi ban tay
                mp_drawing.draw_landmarks(frame, hand_lms,
                                          mp_hands.HAND_CONNECTIONS)

                finger_states = get_finger_states(hand_lms, label)
                raw_gesture = classify_gesture(finger_states)

            # Lam muot: lay cu chi pho bien nhat trong HISTORY_LEN frame
            gesture_history.append(raw_gesture)
            live_gesture = Counter(gesture_history).most_common(1)[0][0]

            now = time.time()
            countdown_value = 0

            # ======================================================
            # MAY TRANG THAI (STATE MACHINE)
            # ======================================================
            if state == COUNTDOWN:
                con_lai = COUNTDOWN_SECONDS - (now - countdown_start)
                if con_lai > 0:
                    # ceil: 2.7s -> hien so 3, 1.2s -> hien so 2, ...
                    countdown_value = int(math.ceil(con_lai))
                else:
                    # HET GIO (moc 0 giay): chot cu chi va may chon ngau nhien
                    locked_player_move = live_gesture
                    computer_move = random.choice(["Rock", "Paper", "Scissors"])

                    if locked_player_move == "Unknown":
                        message = "Chua nhan dien duoc cu chi hop le! Hay thu lai"
                        message_color = YELLOW
                    else:
                        winner = determine_winner(locked_player_move,
                                                  computer_move)
                        if winner == "player":
                            player_score += 1
                            message = "BAN THANG HIEP NAY!"
                            message_color = GREEN
                        elif winner == "computer":
                            computer_score += 1
                            message = "MAY THANG HIEP NAY!"
                            message_color = RED
                        else:
                            message = "HOA!"
                            message_color = YELLOW
                        round_number += 1   # Chi tang hiep khi hiep hop le

                    state = RESULT
                    result_start = now

            elif state == RESULT:
                # Hien ket qua 2.5 giay roi chuyen trang thai
                if now - result_start >= RESULT_SECONDS:
                    if player_score >= WINNING_SCORE:
                        game_over_text = "CHUC MUNG! BAN DA CHIEN THANG CHUNG CUOC!"
                        state = GAME_OVER
                    elif computer_score >= WINNING_SCORE:
                        game_over_text = "MAY TINH CHIEN THANG CHUNG CUOC!"
                        state = GAME_OVER
                    else:
                        state = WAITING

            # ---------- Ve giao dien ----------
            draw_ui(frame, state, player_score, computer_score, round_number,
                    live_gesture, finger_states, locked_player_move,
                    computer_move, countdown_value, message, message_color,
                    game_over_text)

            cv2.imshow("Rock - Paper - Scissors", frame)

            # ---------- Xu ly phim bam ----------
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:        # Q hoac ESC: thoat
                break
            elif key == ord(' ') and state == WAITING:
                state = COUNTDOWN                   # SPACE: bat dau dem nguoc
                countdown_start = time.time()
            elif key == ord('r'):                   # R: reset ve 0 - 0
                player_score = 0
                computer_score = 0
                round_number = 1
                state = WAITING
                gesture_history.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
