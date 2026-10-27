# Bai tap: Tro choi Bua - Keo - Bao qua Camera (MediaPipe Hands + OpenCV)
# Sinh vien: Dang Pham Minh Hoang - MSSV: 26520576

import time
import random
import math
from collections import deque, Counter

import cv2
import mediapipe as mp

# Cau hinh thong so tro choi
WINNING_SCORE = 3
COUNTDOWN_SECONDS = 3
RESULT_SECONDS = 2.5
HISTORY_LEN = 5

# Cac trang thai cua tro choi
WAITING = "WAITING"
COUNTDOWN = "COUNTDOWN"
RESULT = "RESULT"
GAME_OVER = "GAME_OVER"

# Bang mau BGR
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (80, 220, 80)
RED = (60, 60, 240)
YELLOW = (0, 220, 255)
ORANGE = (0, 140, 255)
GRAY = (60, 60, 60)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def get_finger_states(hand_landmarks, handedness_label=None):
    # Xac dinh trang thai 5 ngon tay: [Cai, Tro, Giua, Ap_ut, Ut] (1: duoi, 0: gap)
    lm = hand_landmarks.landmark
    states = []

    # Ngon cai di chuyen ngang -> so sanh khoang cach theo truc X den goc ngon ut (diem 17)
    khoang_cach_dau_ngon = abs(lm[4].x - lm[17].x)
    khoang_cach_khop_ip = abs(lm[3].x - lm[17].x)
    states.append(1 if khoang_cach_dau_ngon > khoang_cach_khop_ip else 0)

    # 4 ngon dai -> so sanh toa do Y cua dau ngon (tip) voi khop PIP ben duoi
    # Trong anh OpenCV, truc Y huong xuong nen diem cao hon co y nho hon
    cac_cap_ngon = [(8, 6), (12, 10), (16, 14), (20, 18)]
    for tip_id, pip_id in cac_cap_ngon:
        states.append(1 if lm[tip_id].y < lm[pip_id].y else 0)

    return states


def classify_gesture(finger_states):
    # Phan loai cu chi dua tren 4 ngon dai (bo qua ngon cai vi de nhieu)
    tro, giua, ap_ut, ut = (
        finger_states[1],
        finger_states[2],
        finger_states[3],
        finger_states[4],
    )

    if tro == 0 and giua == 0 and ap_ut == 0 and ut == 0:
        return "Rock"
    if tro == 1 and giua == 1 and ap_ut == 1 and ut == 1:
        return "Paper"
    if tro == 1 and giua == 1 and ap_ut == 0 and ut == 0:
        return "Scissors"
    return "Unknown"


def determine_winner(player_move, computer_move):
    if player_move == computer_move:
        return "draw"

    cac_truong_hop_thang = {
        ("Rock", "Scissors"),
        ("Scissors", "Paper"),
        ("Paper", "Rock"),
    }
    if (player_move, computer_move) in cac_truong_hop_thang:
        return "player"
    return "computer"


def put_text_center(img, text, center_x, center_y, scale, color, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    org = (int(center_x - w / 2), int(center_y + h / 2))
    cv2.putText(img, text, org, font, scale, BLACK, thickness + 4, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_move_icon(img, move, cx, cy, size, color):
    if move == "Rock":
        cv2.circle(img, (cx, cy), size, color, -1)
        cv2.circle(img, (cx, cy), size, WHITE, 3)
    elif move == "Paper":
        cv2.rectangle(img, (cx - size, cy - size), (cx + size, cy + size), color, -1)
        cv2.rectangle(img, (cx - size, cy - size), (cx + size, cy + size), WHITE, 3)
    elif move == "Scissors":
        cv2.line(img, (cx - size, cy - size), (cx + size, cy + size), color, 12)
        cv2.line(img, (cx - size, cy + size), (cx + size, cy - size), color, 12)
        cv2.circle(img, (cx - size, cy - size), 12, WHITE, 3)
        cv2.circle(img, (cx - size, cy + size), 12, WHITE, 3)
    else:
        put_text_center(img, "?", cx, cy, 2.5, color, 5)


def draw_ui(
    frame,
    state,
    player_score,
    computer_score,
    round_number,
    live_gesture,
    finger_states,
    locked_player_move,
    computer_move,
    countdown_value,
    message,
    message_color,
    game_over_text,
):
    h, w = frame.shape[:2]

    # Thanh trang thai phia tren
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), GRAY, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(
        frame,
        f"Player: {player_score}  -  Computer: {computer_score}",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        WHITE,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"First to {WINNING_SCORE} wins",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        YELLOW,
        2,
        cv2.LINE_AA,
    )

    round_text = f"Hiep: {round_number}"
    (tw, _), _ = cv2.getTextSize(round_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(
        frame,
        round_text,
        (w - tw - 15, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    # Khung hien thi cu chi va mang 5 ngon tay o goc duoi ben trai
    cv2.rectangle(frame, (0, h - 90), (330, h), GRAY, -1)
    cv2.putText(
        frame,
        f"Cu chi: {live_gesture}",
        (12, h - 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        GREEN,
        2,
        cv2.LINE_AA,
    )
    if finger_states is not None:
        mang = "[" + ",".join(str(s) for s in finger_states) + "]"
    else:
        mang = "[ khong thay tay ]"
    cv2.putText(
        frame,
        f"Ngon: {mang}",
        (12, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        WHITE,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "[Cai,Tro,Giua,ApUt,Ut]",
        (190, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        YELLOW,
        1,
        cv2.LINE_AA,
    )

    # Hien thi theo tung trang thai game
    if state == WAITING:
        put_text_center(
            frame,
            "Nhan SPACE de bat dau hiep moi | R: Choi lai | Q: Thoat",
            w // 2,
            h - 120,
            0.7,
            YELLOW,
            2,
        )

    elif state == COUNTDOWN:
        put_text_center(frame, str(countdown_value), w // 2, h // 2, 8, ORANGE, 15)

    elif state == RESULT:
        if locked_player_move in ("Rock", "Paper", "Scissors"):
            put_text_center(frame, "BAN", w // 4, 120, 0.9, WHITE, 2)
            draw_move_icon(frame, locked_player_move, w // 4, 200, 50, GREEN)
            put_text_center(frame, locked_player_move, w // 4, 285, 0.9, GREEN, 2)

            put_text_center(frame, "MAY TINH", 3 * w // 4, 120, 0.9, WHITE, 2)
            draw_move_icon(frame, computer_move, 3 * w // 4, 200, 50, RED)
            put_text_center(frame, computer_move, 3 * w // 4, 285, 0.9, RED, 2)
        put_text_center(frame, message, w // 2, h // 2 + 40, 1.1, message_color, 3)

    elif state == GAME_OVER:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), BLACK, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        put_text_center(frame, game_over_text, w // 2, h // 2 - 40, 1.0, YELLOW, 3)
        put_text_center(
            frame,
            f"Ty so cuoi: {player_score} - {computer_score}",
            w // 2,
            h // 2 + 20,
            0.9,
            WHITE,
            2,
        )
        put_text_center(
            frame,
            "Nhan R de choi van moi | Q / ESC de thoat",
            w // 2,
            h // 2 + 80,
            0.7,
            GREEN,
            2,
        )


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Khong mo duoc camera! Hay kiem tra lai thiet bi.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    state = WAITING
    player_score = 0
    computer_score = 0
    round_number = 1

    countdown_start = 0.0
    result_start = 0.0
    locked_player_move = "Unknown"
    computer_move = "Rock"
    message = ""
    message_color = WHITE
    game_over_text = ""

    gesture_history = deque(maxlen=HISTORY_LEN)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Khong doc duoc khung hinh tu camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            finger_states = None
            raw_gesture = "Unknown"
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                label = None
                if results.multi_handedness:
                    label = results.multi_handedness[0].classification[0].label

                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS
                )

                finger_states = get_finger_states(hand_lms, label)
                raw_gesture = classify_gesture(finger_states)

            # Lay cu chi xuat hien nhieu nhat trong 5 khung hinh gan nhat
            gesture_history.append(raw_gesture)
            live_gesture = Counter(gesture_history).most_common(1)[0][0]

            now = time.time()
            countdown_value = 0

            if state == COUNTDOWN:
                con_lai = COUNTDOWN_SECONDS - (now - countdown_start)
                if con_lai > 0:
                    countdown_value = int(math.ceil(con_lai))
                else:
                    locked_player_move = live_gesture
                    computer_move = random.choice(["Rock", "Paper", "Scissors"])

                    if locked_player_move == "Unknown":
                        message = "Chua nhan dien duoc cu chi hop le! Hay thu lai"
                        message_color = YELLOW
                    else:
                        winner = determine_winner(locked_player_move, computer_move)
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
                        round_number += 1

                    state = RESULT
                    result_start = now

            elif state == RESULT:
                if now - result_start >= RESULT_SECONDS:
                    if player_score >= WINNING_SCORE:
                        game_over_text = "CHUC MUNG! BAN DA CHIEN THANG CHUNG CUOC!"
                        state = GAME_OVER
                    elif computer_score >= WINNING_SCORE:
                        game_over_text = "MAY TINH CHIEN THANG CHUNG CUOC!"
                        state = GAME_OVER
                    else:
                        state = WAITING

            draw_ui(
                frame,
                state,
                player_score,
                computer_score,
                round_number,
                live_gesture,
                finger_states,
                locked_player_move,
                computer_move,
                countdown_value,
                message,
                message_color,
                game_over_text,
            )

            cv2.imshow("Rock - Paper - Scissors", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" ") and state == WAITING:
                state = COUNTDOWN
                countdown_start = time.time()
            elif key == ord("r"):
                player_score = 0
                computer_score = 0
                round_number = 1
                state = WAITING
                gesture_history.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
