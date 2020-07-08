import numpy as np
import os
import cv2
import random
from math import sqrt
import skimage
from matplotlib import pyplot as plt
from scipy.stats import skew
from matplotlib.pyplot import plot, xlim

def plot_hist(img):
    histo, x = np.histogram(img, range(0, 256), density=True)
    plot(histo)
    xlim(0, 255)

def convert_to_gray(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return imgray


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

def apply_threshold(img, t):
    _, thresh = cv2.threshold(img, t, 255, 0)
    thresh = cv2.bitwise_not(thresh)
    return thresh


def apply_closing_morphology(img, matrix_size):
    kernel = np.ones((matrix_size, matrix_size), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing


def apply_opening_morphology(img, matrix_size):
    kernel = np.ones((matrix_size, matrix_size), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening


def apply_dilation_morphology(img, matrix_size, iterations):
    kernel = np.ones((matrix_size, matrix_size), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=iterations)
    return dilation


def load_easy_images():
    images = []
    for i in range(8):
        path = os.path.join('easy_set', str(i + 1) + '.jpg')
        images.append(cv2.imread(path))
    return images


def load_medium_images():
    images = []
    for i in range(10):
        path = os.path.join('medium_set', str(i + 1) + '.jpg')
        images.append(cv2.imread(path))
    return images

def load_hard_images():
    images = []
    for i in range(9):
        path = os.path.join('hard_set', str(i + 1) + '.jpg')
        images.append(cv2.imread(path))
    return images

def find_game_boards_from_contours(contours):
    boards = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        boards.append(Box(i, x, y, w, h, contours[i]))
    remove_unnecessary_contours(boards)
    return boards


def remove_unnecessary_contours(game_boards):
    max_area = max(box.getArea() for box in game_boards)
    to_remove = []
    for i in range(len(game_boards)):
        for j in range(len(game_boards)):
            if j not in to_remove:
                if game_boards[i].checkIfInside(game_boards[j]) or game_boards[j].getArea() / max_area <= 0.05:
                    to_remove.append(j)

    to_remove = sorted(to_remove, reverse=True)
    for x in to_remove:
        del game_boards[x]


def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_convex_hull_points(contours):
    points = cv2.convexHull(contours)
    to_remove = []
    for i in range(len(points) - 1):
        if i not in to_remove:
            for j in range(i, len(points)):
                if i != j and distance(points[i][0], points[j][0]) < game.distance:
                    to_remove.append(j)

    to_remove = sorted(list(set(to_remove)), reverse=True)
    for x in to_remove:
        points = np.delete(points, x, 0)
    return points


def build_pairs_from(points):
    if distance(points[0], points[1]) > distance(points[1], points[2]):
        points = [points[-1]] + points[:-1]
    pairs = []
    for x in range(0, 8, 2):
        pairs.append(points[x:x + 2])
    return pairs


def pairs_have_correct_no_points(pairs):
    return all(len(pair) == 2 for pair in pairs)


def get_game_board_points(pairs):
    return np.float32([(pairs[i][0] + pairs[i][1]) /
                       2 for i in range(len(pairs))])


def intersection(rho1, theta1, rho2, theta2):
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def analyze_field_shapes(img, gameindex, i, j):
    w, h = img.shape
    center = [w // 2, h // 2]

    img = apply_dilation_morphology(img, 5, 1)

    for k in range(h):
        if np.count_nonzero(img[:, k]) > 0.8 * w:
            img[:, k] = 0
    for k in range(w):
        if np.count_nonzero(img[k, :]) > 0.8 * h:
            img[k, :] = 0

    conts, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.cv2.CHAIN_APPROX_NONE)

    # conts = sorted(conts, key=lambda x: cv2.boundingRect(x)[2] * cv2.boundingRect(x)[3], reverse=True)

    test = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if (len(conts)):

        newconts = [[cont, np.min([distance(center, point.ravel()) for point in cont])] for cont in conts]
        newconts = sorted(newconts, key=lambda x: x[1])

        flag=0
        for x in newconts:
            if x[1] > 0.25 * w:              # jesli odleglosc najblizszego punktu od srodka wieksza od 0.3 szerokosci to pole puste
                break
            if cv2.boundingRect(x[0])[2] * cv2.boundingRect(x[0])[3] > 0.1 * w * h: #jesli pole prostokata wieksze od 0.1 pola pola przejdz dalej
                shape=x[0]
                flag=1
                break
        if not flag:    #syf zabezpieczajacy przed nie wykryciem niczego co by bylo wystarczajace duze, a zaden element nie wywalil
            # cv2.drawContours(test, x[0], -1, (0, 255, 0), 2)
            # cv2.imshow(str(gameindex) + '-' + str(i) + str(j), cv2.resize(test, (200, 200)))
            return ' '

        # cv2.drawContours(test, shape, -1, (0, 255, 0), 2)

        cx = 0
        cy = 0
        for p in shape:
            cx += p[0][0]
            cy += p[0][1]
        cx = int(cx / len(shape))
        cy = int(cy / len(shape))

        area = cv2.contourArea(shape)
        if area:
            # cv2.circle(test, (cx, cy), 3, (0, 0, 255), -3)
            # cv2.imshow(str(gameindex) + '-' + str(i) + str(j), cv2.resize(test, (200, 200)))
            hull = cv2.convexHull(shape)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area
            if (solidity > 0.5):
                return 'o'
            else:
                mask = np.zeros(img.shape,np.uint8)
                cv2.drawContours(mask,[shape],0,255,-1)
                # cv2.imshow("mask - " + str(gameindex) + str(i) + str(j), mask)
                if mask[cy][cx] == 0:
                    return 'o'
                else:
                    return 'x'

    # cv2.imshow(str(gameindex) + '-' + str(i) + str(j), cv2.resize(test, (200, 200)))
    return ' '


def analyze_single_board(img, index):
    original_image = img
    img = convert_to_gray(img)
    img = apply_threshold(img, 125)
    img = apply_closing_morphology(img, 0)

    edges = cv2.Canny(img, 100, 100, apertureSize=3)
    h_lines = cv2.HoughLines(edges, 1, np.pi / 180, 90)

    horizontal_lines = []
    vertical_lines = []
    if h_lines is not None:
        for line in h_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            if abs(theta - np.pi) < 0.1 or abs(theta - np.pi / 2) < 0.1 or abs(theta) < 0.1:
                if abs(y2 - y1) < abs(x2 - x1):
                    horizontal_lines.append([x1, x2, y1, y2, rho, theta])
                elif abs(x2 - x1) < abs(y2 - y1):
                    vertical_lines.append([x1, x2, y1, y2, rho, theta])

        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            max_y_distance = 0
            line1 = horizontal_lines[0]
            line2 = horizontal_lines[1]
            for i in range(len(horizontal_lines)):
                for j in range(i + 1, len(horizontal_lines)):
                    line_i = horizontal_lines[i]
                    line_j = horizontal_lines[j]
                    y_distance = abs(line_i[2] - line_j[2]) + abs(line_i[3] - line_j[3])
                    if y_distance > max_y_distance:
                        max_y_distance = y_distance
                        line1 = line_i
                        line2 = line_j

            max_x_distance = 0
            line3 = vertical_lines[0]
            line4 = vertical_lines[1]
            for i in range(len(vertical_lines)):
                for j in range(i + 1, len(vertical_lines)):
                    line_i = vertical_lines[i]
                    line_j = vertical_lines[j]
                    x_distance = abs(line_i[0] - line_j[0]) + abs(line_i[1] - line_j[1])
                    if x_distance > max_x_distance:
                        max_x_distance = x_distance
                        line3 = line_i
                        line4 = line_j

            points = []
            points += intersection(line1[4], line1[5], line3[4], line3[5])
            points += intersection(line1[4], line1[5], line4[4], line4[5])
            points += intersection(line2[4], line2[5], line3[4], line3[5])
            points += intersection(line2[4], line2[5], line4[4], line4[5])

            # for point in points:
            #     cv2.circle(original_image, (point[0], point[1]), 5, (0, 255, 255))

            # cv2.line(original_image, (line1[0], line1[2]),
            #         (line1[1], line1[3]), (0, 0, 255), 1)
            # cv2.line(original_image, (line2[0], line2[2]),
            #         (line2[1], line2[3]), (0, 0, 255), 1)
            # cv2.line(original_image, (line3[0], line3[2]),
            #         (line3[1], line3[3]), (0, 0, 255), 1)
            # cv2.line(original_image, (line4[0], line4[2]),
            #         (line4[1], line4[3]), (0, 0, 255), 1)

            test = points
            test = [x[::-1] for x in test]
            h = np.mean([x[0] for x in test])
            t1 = [x for x in test if x[0] < h]
            t2 = [x for x in test if x[0] > h]
            t1 = sorted(t1, key=lambda x: x[1])
            t2 = sorted(t2, key=lambda x: x[1])
            test = t1 + t2

            sectors = [[[0, 0], [0, test[0][1]], [0, test[1][1]], [0, 300]],
                    [[test[0][0], 0], test[0], test[1], [test[1][0], 300]],
                    [[test[2][0], 0], test[2], test[3], [test[3][0], 300]],
                    [[300, 0], [300, test[2][1]], [300, test[3][1]], [300, 300]]]

            sectors = [[[0, 0], [0, test[0][1]], [0, test[1][1]], [0, 300]],
                    [[test[0][0], 0], test[0], test[1], [test[1][0], 300]],
                    [[test[2][0], 0], test[2], test[3], [test[3][0], 300]],
                    [[300, 0], [300, test[2][1]], [300, test[3][1]], [300, 300]]]

            cut = 0
            tab = [[' '] * 3 for i in range(3)]
            for i in range(3):
                for j in range(3):
                    if game.index==12 and i==2 and j==1:
                        sectors[i][j + 1][1]=199
                        cv2.imshow('wazne', img[sectors[i][j][0] + cut:sectors[i + 1][j][0] - cut,
                                            sectors[i][j][1] + cut:sectors[i][j + 1][1] - cut])

                    field = cv2.resize(img[sectors[i][j][0] + cut:sectors[i + 1][j][0] - cut,
                            sectors[i][j][1] + cut:sectors[i][j + 1][1] - cut], (200, 200))
                    tab[i][j] = analyze_field_shapes(field, game.index, i, j)

                    # cv2.imshow(str(index) + '-' + str(i) + str(j), cv2.resize(field, (200, 200)))
            print('plansza ' + str(game.index))
            for x in tab:
                print(x)

            bottomLeftCornerOfText = (10, 20)
            for i in range(len(tab)):
                cv2.putText(original_image, tab[i][0] + tab[i][1] + tab[i][2], bottomLeftCornerOfText, font,
                            fontScale, fontColor, lineType)
                a, b = bottomLeftCornerOfText
                b += 20
                bottomLeftCornerOfText = (a, b)
            cv2.imshow(str(index), original_image)




class Box:
    def __init__(self, index, x, y, w, h, contour):
        self.index = index
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.point = np.array([x, y])
        self.center = np.array([x + w / 2, y + h / 2])
        self.shape = np.array([x + w, y + h])
        self.distance = 0.1 * w
        self.contour = contour
        M = cv2.moments(contour)
        try:
            self.centroid = (int(M['m10'] / M['m00']),
                             int(M['m01'] / M['m00']))
        except:
            self.centroid = (0, 0)

    def checkIfInside(self, otherBox):
        if self.index == otherBox.index:
            return False
        return all(self.point < otherBox.center) and all(self.shape > otherBox.center)

    def getPoint(self):
        return self.point

    def getShape(self):
        return self.shape

    def getArea(self):
        return abs(self.getPoint()[0] - self.getShape()[0]) * abs(self.getPoint()[1] - self.getShape()[1])

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

easy_images = load_easy_images()
medium_images = load_medium_images()
hard_images = load_hard_images()

original_image = easy_images[2]

img = convert_to_gray(original_image)

img = apply_threshold(img, 125)

img = apply_closing_morphology(img, 2)

cv2.imshow("img", img)

contours, _ = cv2.findContours(
    img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

game_boards = find_game_boards_from_contours(contours)

for i in range(len(game_boards)):
    game = game_boards[i]
    points = get_convex_hull_points(contours[game.index])
    flattened_points = [points[i].ravel() for i in range(len(points))]
    pairs = build_pairs_from(flattened_points)

    if pairs_have_correct_no_points(pairs):
        # for pair in pairs:
            # cv2.circle(original_image, (pair[0][0], pair[0][1]), 10, (0, 0, 255), -3)
            # cv2.circle(original_image, (pair[1][0], pair[1][1]), 10, (0, 0, 255), -3)

        pts1 = get_game_board_points(pairs)

        # for point in pts1:
        #     cv2.circle(original_image, (point[0], point[1]), 10, (255, 0, 0), -3)

        pts2 = np.float32([[300, 150], [150, 300], [0, 150], [150, 0]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(original_image, M, (300, 300), borderValue=(255, 255, 255))
        # cv2.imshow('dst' + str(game.index), dst)
        analyze_single_board(dst, game.index)

cv2.imshow('res', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
