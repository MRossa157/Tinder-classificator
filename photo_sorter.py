import shutil
import os
import json
import cv2
import matplotlib.pyplot as plt
import random


def validator(image_path, debug=False):
    # Загружаем каскадный классификатор
    cascade_path = r"C:\Users\PC\AppData\Local\Programs\Python\Python39\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Импортируем изображение в cv2
    image_filename = image_path[image_path.rfind("\\") + 1:]
    image_path = image_path[:image_path.rfind("\\")]
    normalize_image_path = os.path.join(image_path, image_filename)
    image = cv2.imread(image_path)

    # Если оно криво импортировалось(что происходит почти всегда), тогда используем костыль ниже
    if image is None:
        shutil.copy(normalize_image_path, image_filename)
        image = cv2.imread(image_filename)
        os.remove(image_filename)

    # Переводим изображение в черно-белый формат
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Определяем области где есть лица
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minNeighbors=4)
    if debug:
        faces_detected = "Лиц обнаружено: " + format(len(faces))
        print(faces_detected)
        # Если лицо нашлось добавляем выводим его
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Показываем фото и ждем нажатия ESC
        cv2.imshow("image", image)
        cv2.waitKey()
    return len(faces)


def photo_sort(flag_validator=False):
    dataset_path = r"C:\Users\PC\Desktop\Проги\ML\Tinder_bot\Dataset\Unsorted"
    if flag_validator:
        print("Валидатор лиц: Включен")
    else:
        print("Валидатор лиц: Выключен")

    with open(rf"{dataset_path}\result.json", "rb") as read_file:
        data = json.load(read_file)

    print("Создали папки для сортировки")
    if not os.path.isdir(dataset_path + r"\Like"):
        os.mkdir(dataset_path + r"\Like")

    if not os.path.isdir(dataset_path + r"\Dislike"):
        os.mkdir(dataset_path + r"\Dislike")

    # Сортируем фотографии по папкам Like и Dislike
    print("Открыли файл и начали сортировку")
    reactions = ["💌", "❤️", "👎"]
    for item in data["messages"]:
        if item["from"] == "Дайвинчик | Leomatchbot":
            try:
                photo = item["photo"]  # Сохраняем фото если оно есть
            except Exception as e:
                pass
        else:
            # Если в сообщении есть реакция, то это ТВОЕ сообщение -> можно относить это сообщение к сохраненной фотке
            if item["text"] in reactions:
                try:
                    if flag_validator:
                        if (item["text"] == reactions[0]
                            or item["text"] == reactions[1]
                            ) and validator(rf"{dataset_path}\{photo}"):
                            shutil.copy(rf"{dataset_path}\{photo}",
                                        rf"{dataset_path}\Like")

                        elif item["text"] == reactions[2] and validator(
                                rf"{dataset_path}\{photo}"):
                            shutil.copy(rf"{dataset_path}\{photo}",
                                        rf"{dataset_path}\Dislike")

                    else:
                        if item["text"] == reactions[0] or item[
                                "text"] == reactions[1]:
                            shutil.copy(rf"{dataset_path}\{photo}",
                                        rf"{dataset_path}\Like")

                        elif item["text"] == reactions[2]:
                            shutil.copy(rf"{dataset_path}\{photo}",
                                        rf"{dataset_path}\Dislike")
                except:
                    pass


def main():
    print("Начали работу")
    for photo in os.listdir(r"C:\Users\PC\Desktop\Проги\ML\Tinder_bot\Dataset\Dislike"):
        img = os.path.join(
            r"C:\Users\PC\Desktop\Проги\ML\Tinder_bot\Dataset\Dislike", photo)
        print(validator(img, True))
        break


if __name__ == "__main__":
    main()
