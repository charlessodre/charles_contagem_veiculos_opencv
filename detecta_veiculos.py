import cv2
import time

print(cv2.__version__)

video_entrada = "./videos/Cars - 133.mp4"

# Acessa o video que será analisado.
cap = cv2.VideoCapture(video_entrada)
_, frame = cap.read()

# Define as configurações do vídeo de saida que vai conter os pontos detectados e movimentos detectados.
output_frame = "./output/saida_video.mp4"
save_frame = cv2.VideoWriter(output_frame, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame.shape[1], frame.shape[0]))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

total_left_car = 0
total_right_car = 0

# Area mínima do contorno considerado.
min_contour_Area = 20000
min_center_distance = 60
gaussian_blur_value = 3
threshold_binary_value = 5
y_current_position = None
x_current_position = None

while True:

    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()

    if not ret1 or not ret2:
        break

    time.sleep(0.2)

    frame_width = frame1.shape[1]
    frame_height = frame1.shape[0]

    # Converte os frames para cinza.
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Aplica o GaussianBlur- Smoothing Images
    frame1_gray_blur = cv2.GaussianBlur(frame1_gray, (gaussian_blur_value, gaussian_blur_value), 0)
    frame2_gray_blur = cv2.GaussianBlur(frame2_gray, (gaussian_blur_value, gaussian_blur_value), 0)

    # Obtêm as diferenças entre os frames.
    frame_diff = cv2.absdiff(frame1_gray_blur, frame2_gray_blur)

    # Aplica o "Simple Thresholding" para tentar destacar a imagem principal (sem o fundo).
    _, frame1_gray_blur_binary = cv2.threshold(frame_diff, threshold_binary_value, 255, cv2.THRESH_BINARY)

    # Aumenta a área da imagem após a aplicação do Image Thresholding.
    frame_dilate = cv2.dilate(frame1_gray_blur_binary, None, iterations=2)

    # Extrai o contorno da imagem analisada.
    contours, _ = cv2.findContours(frame_dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenha o contorno na imagem
    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2) # thickness=cv2.FILLED)

    y_split_lanes = 230
    cv2.line(frame1, (0, y_split_lanes), (frame_width, y_split_lanes), (255, 0, 0), 2)

    x_target_line = int(frame_width / 2)
    cv2.line(frame1, (x_target_line, 0), (x_target_line, frame_height), (255, 0, 0), 2)

    cv2.putText(frame1, "Pista Esquerda", (400, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    cv2.putText(frame1, "Pista Direita", (400, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    for cnt in contours:

        # Obtêm a área do contorno.
        contour_area = cv2.contourArea(cnt)

        if contour_area > min_contour_Area:

            (x, y, w, h) = cv2.boundingRect(cnt)

            cv2.rectangle(frame1, (x, y), (x + w, y + h), (85, 85, 255), 2)

            x_center = int((x + x + w) / 2)
            y_center = int((y + y + h) / 2)
            center_point = (x_center, y_center)
            # cv2.circle(frame1, center_point, 10, (0, 0, 255), thickness=-1)
            # cv2.putText(frame1, "{}".format(str(center_point)), center_point, cv2.FONT_HERSHEY_PLAIN, 1, (255,0 , 0), 1, cv2.LINE_AA)
            # cv2.putText(frame1, "{}".format(str(areaContorno)), center_point, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)

            y_current_position = y_center
            x_current_position = x_center

            current_coord = center_point

            abs_distance_target = abs(x_current_position - x_target_line)

            if y_current_position < y_split_lanes:  # Pista da esquerda
                cv2.putText(frame1, "Esquerda", current_coord, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                if x_current_position > x_target_line:
                    if abs_distance_target < min_center_distance:
                        total_left_car += 1
                        break

            else:  # Pista da direita
                cv2.putText(frame1, "Direita", current_coord, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                if x_current_position > x_target_line:
                    if abs_distance_target < min_center_distance:
                        total_right_car += 1
                        break

    cv2.putText(frame1, "Carros Pista Esquerda: {}".format(str(total_left_car)), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 0), 1)
    cv2.putText(frame1, "Carros Pista Direita: {}".format(str(total_right_car)), (5, 35), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 0), 1)
    cv2.putText(frame1, "Carros Total: {}".format(str(total_right_car + total_left_car)), (5, 50),
                cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 0), 1)

    # Exibe a saida Video
    # cv2.imshow('frame_gray - 1', frame1_gray)
    # cv2.imshow('frame_gray_blur - 2', frame1_gray_blur)
    # cv2.imshow('frame_gray_blur_binary - 3', frame1_gray_blur_binary)
    # cv2.imshow('frame_diff - 4', frame_diff)
    cv2.imshow('frame_dilate - 5', frame_dilate)
    cv2.imshow('original', frame1)

    # Salva uma cópia do frame
    save_frame.write(frame1)

    # Aperte a tecla 'q' para sair.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos.
save_frame.release()
cv2.destroyAllWindows()
