import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from tkinter import filedialog
from tkinter import *
from PIL import Image
from scipy import ndimage
import copy
import scipy.interpolate as si
import math
import xlsxwriter
import statistics


def load_images_from_folder(folder):
    loaded_images = []
    loaded_folder = os.chdir(folder)
    loaded_folder = sorted(filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime)
    for file_name in loaded_folder:
        new_path = folder + '/' + file_name
        if new_path.find('Thumbs.db') == -1:
            image = np.array(Image.open(new_path))
            red_channel, green_channel, blue_channel = cv2.split(image)
            image = cv2.merge((blue_channel, green_channel, red_channel))
            if image is not None:
                loaded_images.append(image)
    return loaded_images, loaded_folder


def load_control_image(control_path):
    control_image = np.array(Image.open(control_path))
    red_control, green_control, blue_control = cv2.split(control_image)
    control_image = cv2.merge((blue_control, green_control, red_control))
    return red_control


def calculate_values(N, M, mask, counter, plot_values, chan, image_value):
    for j in range(1, N + 1, mask):
        if plot_values == 'mean':
            stripe = chan[counter:j, 0:M]
            mean_value = np.mean(stripe)
            image_value.append(round(mean_value, 2))
        elif plot_values == 'sum':
            stripe = chan[counter:j, 0:M]
            sum_value = np.sum(stripe)
            image_value.append(round(sum_value, 2))
        counter = counter + mask
    return image_value


def control_val(control_image, r, plot_values, second_roi, second_roi_width):
    # if second_roi == "no":
    #     control_roi = control_image[(r[1]):(r[1] + r[3]), (r[0]):(r[0] + r[2])]
    # elif second_roi == "yes":
    #     control_roi = control_image[(r[1]):(r[1] + r[3]), (r[0]):(r[0] + r[2])]
    #     N, M = control_roi.shape[:2]
    #     control_roi = control_roi[0:N, 0:second_roi_width]
    # N_control, M_control = control_roi.shape[:2]
    # control_values = []
    # control_values = calculate_values(N_control, M_control, 1, 0, plot_values, control_roi, control_values)
    # # control_values = ndimage.median_filter(control_values, size=median_filter_size)

    if second_roi == "no":
        r_k = (round(326-(r[2]/2)), 150, r[2], r[3])
        control_roi = control_image[(r_k[1]):(r_k[1] + r_k[3]), (r_k[0]):(r_k[0] + r_k[2])]
    elif second_roi == "yes":
        r_k = (round(326 - (r[2] / 2)), 150, r[2], r[3])
        control_roi = control_image[(r_k[1]):(r_k[1] + r_k[3]), (r_k[0]):(r_k[0] + r_k[2])]
        N, M = control_roi.shape[:2]
        control_roi = control_roi[0:N, 0:second_roi_width]
    N_control, M_control = control_roi.shape[:2]
    control_values = []
    control_values = calculate_values(N_control, M_control, 1, 0, plot_values, control_roi, control_values)

    max_value = max(control_values)
    for l in range(0, len(control_values)):
        control_values[l] = control_values[l] / max_value
    control_roi = cv2.normalize(control_roi, None, alpha=0.01, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return control_values, control_roi


def normalization(mean_values, N):
    x = [m for m in range(0, N)]

    control_model = [1.19124435e-07, -7.94288954e-05, 1.31339232e-02, 1.81886356e-01]
    control_values = np.polyval(control_model, x)
    # control_values = [0.4822222476014848, 0.4707728509563136, 0.4601274145449041, 0.4502645618341207,
    # 0.44116320686868804,
    #  0.4328025525470285, 0.4251620888971024, 0.4182215913522465, 0.41196111902701354, 0.40636101299301136,
    #  0.4014018945547413, 0.3970646635254384, 0.39333049650290985, 0.39018084514537443, 0.3875974344473012,
    #  0.3855622610152492, 0.3840575913437061, 0.3830659600909277, 0.3825701683547766, 0.3825532819485623,
    #  0.38299862967687887, 0.3838898016114452, 0.3852106473669437, 0.38694527437685955, 0.38907804616931996,
    #  0.3915935806429326, 0.3944767483426258, 0.39771267073548694, 0.4012867184866017, 0.4051845097348931,
    #  0.4093919083689614, 0.4138950223029218, 0.41868020175224485, 0.4237340375095951, 0.4290433592206701,
    #  0.43459523366003966, 0.44037696300698503, 0.44637608312133803, 0.4525803618193202, 0.45897779714938136,
    #  0.46555661566803963, 0.4723052707157204, 0.4792124406925947, 0.48626702733441896, 0.4934581539883745,
    #  0.5007751638889052, 0.5082076184335587, 0.5157452954588237, 0.5233781875159701, 0.5310965001468878,
    #  0.5388906501599257, 0.5467512639057316, 0.5546691755530898, 0.5626354253647619, 0.5706412579733249,
    #  0.5786781206570107, 0.5867376616155451, 0.594811728245987, 0.6028923654185673, 0.610971813752529,
    #  0.619042507891964, 0.6270970747816554, 0.6351283319429138, 0.643129285749419, 0.6510931297030559,
    #  0.6590132427097575, 0.6668831873553402, 0.6746967081813461, 0.6824477299608803, 0.6901303559744503,
    #  0.6977388662858052, 0.7052677160177756, 0.712711533628112, 0.7200651191853232, 0.7273234426445171,
    #  0.7344816421232383, 0.7415350221773089, 0.7484790520766654, 0.7553093640811999, 0.7620217517165981,
    #  0.7686121680501786, 0.7750767239667323, 0.7814116864443612, 0.787613476830318, 0.7936786691168446,
    #  0.799603988217012, 0.8053863082405581, 0.8110226507697287, 0.8165101831351149, 0.8218462166914934,
    #  0.8270282050936646, 0.832053742572293, 0.8369205622097456, 0.8416265342159304, 0.8461696642041371,
    #  0.8505480914668747, 0.8547600872517108, 0.8588040530371124, 0.8626785188082835, 0.8663821413330034,
    #  0.8699137024374681, 0.873272107282128, 0.8764563826375273, 0.8794656751601431, 0.8822992496682246,
    #  0.8849564874176328, 0.887436884377677, 0.8897400495069584, 0.8918657030292059, 0.8938136747091155,
    #  0.8955839021281906, 0.8971764289605811, 0.8985914032489204, 0.8998290756801673, 0.9008897978614445,
    #  0.9017740205958755, 0.9024822921584271, 0.9030152565717459, 0.9033736518819984, 0.9035583084347093,
    #  0.903570147150604, 0.9034101778014425, 0.9030794972858626, 0.9025792879052164, 0.9019108156394117,
    #  0.9010754284227503, 0.9000745544197657, 0.8989097003010644, 0.8975824495191632, 0.89609446058433,
    #  0.8944474653404216, 0.8926432672407235, 0.8906837396237884, 0.8885708239892767, 0.886306528273794,
    #  0.8838929251267309, 0.8813321501861026, 0.8786264003543874, 0.8757779320743655, 0.8727890596049598,
    #  0.8696621532970724, 0.8663996378694265, 0.8630039906844036, 0.8594777400238833, 0.8558234633650825,
    #  0.852043785656394, 0.8481413775932272, 0.8441189538938442, 0.8399792715752028, 0.8357251282287919,
    #  0.8313593602964735, 0.8268848413463201, 0.8223044803484549, 0.8176212199508898, 0.8128380347553654,
    #  0.8079579295931906, 0.8029839378010795, 0.7979191194969935, 0.792766559855978, 0.7875293673860034,
    #  0.7822106722038025, 0.7768136243107105, 0.7713413918685049, 0.765797159475243, 0.7601841264411017,
    #  0.7545055050642179, 0.748764518906525, 0.7429644010695949, 0.7371083924704745, 0.7311997401175272,
    #  0.7252416953862705, 0.7192375122952147, 0.7131904457817039, 0.7071037499777542, 0.7009806764858917,
    #  0.6948244726549935, 0.6886383798561257, 0.6824256317583826, 0.6761894526047264, 0.6699330554878254,
    #  0.6636596406258947, 0.6573723936385328, 0.6510744838225635, 0.6447690624278737, 0.6384592609332519,
    #  0.6321481893222283, 0.6258389343589134, 0.6195345578638385, 0.6132380949897919, 0.6069525524976616,
    #  0.6006809070322712, 0.5944261033982217, 0.5881910528357288, 0.5819786312964623, 0.5757916777193862,
    #  0.5696329923065971, 0.563505334799163, 0.5574114227529634, 0.5513539298145274, 0.5453354839968736,
    #  0.5393586659553488, 0.5334260072634672, 0.5275399886887503, 0.5217030384685639, 0.5159175305859602,
    #  0.5101857830455141, 0.5045100561491644, 0.49889255077205175, 0.49333540663835856, 0.4878407005971469,
    #  0.4824104448981995, 0.47704658546785716, 0.4717510001848585, 0.46652549715617964, 0.46137181299287233,
    #  0.4562916110859039, 0.45128647988199583, 0.44635793115946326, 0.44150739830405383, 0.43673623458478705,
    #  0.4320457114297933, 0.427437016702153, 0.4229112529757355, 0.4184694358110387, 0.4141124920310279,
    #  0.4098412579969747, 0.40565647788429654, 0.40155880195839533, 0.3975487848504975, 0.393626883833492,
    #  0.3897934570977701, 0.38604876202706434, 0.38239295347428776, 0.37882608203737295, 0.375348092335111,
    #  0.37195882128299107, 0.3686579963690389, 0.3654452339296568, 0.3623200374254617, 0.3592817957171252,
    #  0.3563297813412124, 0.3534631487860208, 0.3506809327674192, 0.34798204650468834, 0.3453652799963578,
    #  0.3428292982960468, 0.3403726397883026, 0.33799371446443993, 0.3356908021983797, 0.3334620510224887,
    #  0.33130547540341854, 0.32921895451794425, 0.3272002305288042, 0.32524690686053864, 0.32335644647532924,
    #  0.3215261701488378, 0.31975325474604593, 0.3180347314970936, 0.31636748427311845, 0.31474824786209543,
    #  0.313173606244675, 0.3116399908700228, 0.3101436789316592, 0.3086807916432973, 0.3072472925146833,
    #  0.3058389856274345, 0.30445151391087927, 0.30308035741789596, 0.3017208316007517, 0.30036808558694184,
    #  0.29901710045502916, 0.29766268751048264, 0.2962994865615168, 0.29492196419493094, 0.2935244120519483,
    #  0.2921009451040545, 0.2906454999288377, 0.2891518329858269, 0.28761351889233194, 0.28602394869928127,
    #  0.2843763281670625, 0.2826636760413608, 0.28087882232899797, 0.27901440657377197, 0.27706287613229563,
    #  0.2750164844498361, 0.2728672893361538, 0.27060715124134166, 0.2682277315316641, 0.2657204907653962,
    #  0.263076686968663, 0.26028737391127854, 0.25734339938258466, 0.2542354034672907, 0.25095381682131224,
    #  0.2474888589476103, 0.2438305364720305, 0.23996864141914226, 0.23589274948807779]

    if len(control_values) != len(mean_values):
        to_add = len(mean_values)-len(control_values)
        for i in range(0, to_add):
            control_values.append(control_values[-1])

    for k in range(0, len(mean_values)):
        mean_values[k] = mean_values[k]/control_values[k]

    return mean_values, control_values


def rotation_angle(images):
    r = cv2.selectROI(images[0])
    cv2.destroyAllWindows()
    roi_area = images[0][(r[1]):(r[1] + r[3]), (r[0]):(r[0] + r[2])]
    N, M = roi_area.shape[:2]
    x_coordinates = []
    y_coordinates = []
    stripe_counter = 0
    for j in range(1, N + 1):
        stripe = roi_area[stripe_counter:j, 0:M, 2][0]
        indexes_of_max_values = np.where(stripe == np.max(stripe))[0]
        center_element = int(len(indexes_of_max_values) / 2)
        x_coordinates.append(indexes_of_max_values[center_element])
        y_coordinates.append(j)
        stripe_counter = stripe_counter + 1

    # for k in range(1, len(x_coordinates) - 1):
    #     roi_area[y_coordinates[k]][x_coordinates[k]][0] = 255
    #     roi_area[y_coordinates[k]][x_coordinates[k]][1] = 0
    #     roi_area[y_coordinates[k]][x_coordinates[k]][2] = 0

    curve_approximation = np.polyfit(x_coordinates, y_coordinates, 1)
    tile_angle = curve_approximation[0]

    image_center = tuple(np.array(roi_area.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -tile_angle / 2, 1.0)
    result = cv2.warpAffine(roi_area, rot_mat, roi_area.shape[1::-1], flags=cv2.INTER_LINEAR)
    cv2.imshow('test', result)
    cv2.waitKey(0)
    return tile_angle


def image_rotation(image, tile_angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -tile_angle / 2, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def camera_calibration(img):
    K = np.array([[6.666920265485503023e+02, 0.000000000000000000e+00, 3.197177260187415300e+02],
                  [0.000000000000000000e+00, 6.655078053495720951e+02, 2.234531438841754607e+02],
                  [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
    d = np.array([8.074266592516801677e-02, -8.680594598066510015e-01, 1.736000787309049248e-03,
                  -3.093798423381155689e-03, 2.535723708587016567e+00])
    h, w = img.shape[:2]
    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 0)
    mapx, mapy = cv2.initUndistortRectifyMap(K, d, None, newcameramatrix, (w, h), 5)

    # Remap the original image to a new image
    newimg = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    red_channel2, green_channel2, blue_channel2 = cv2.split(newimg)
    newimg = cv2.merge((blue_channel2, green_channel2, red_channel2))
    return newimg


def divide_into_regions(chan):
    N, M = chan.shape[:2]
    number_of_regions = 5
    width_of_regions = round(M / number_of_regions)
    region_images = []
    tmp_r0 = 0
    central_region = math.ceil(number_of_regions/2)-1
    for c in range(0, number_of_regions):
        region_images.append(chan[0:N, tmp_r0:(tmp_r0 + width_of_regions)])
        tmp_r0 = tmp_r0 + width_of_regions
        if tmp_r0 + width_of_regions > M:
            width_of_regions = width_of_regions + (M - (width_of_regions * number_of_regions))
    return region_images, central_region


def regions_to_draw(r, n_r):
    number_of_regions = 5
    width_of_regions = round(r[2] / number_of_regions)
    new_r = []
    tmp_r0 = n_r[0]
    for c in range(0, number_of_regions):
        new_r.append([tmp_r0, n_r[1], width_of_regions, n_r[3]])
        tmp_r0 = tmp_r0 + width_of_regions
        if tmp_r0 + width_of_regions > (n_r[0] + n_r[2]):
            width_of_regions = width_of_regions + (n_r[2] - (width_of_regions * number_of_regions))
    return new_r


def plot_derivatives(new_chan, roi_area, roi_area_to_plot, derivatives1, derivatives2, derivatives3,
                     greatest_derivative1, greatest_derivative2, greatest_derivative3, max_derivative1, max_derivative2,
                     max_derivative3, x1, x2, x3, y1, y2, y3, time, i, thickness, save_path):
    img1 = copy.deepcopy(new_chan)
    img1 = cv2.applyColorMap(img1, cv2.COLORMAP_VIRIDIS)
    r_img1, g_img1, b_img1 = cv2.split(img1)
    img1 = cv2.merge([b_img1, g_img1, r_img1], 3)

    img2 = copy.deepcopy(roi_area)
    img2 = cv2.applyColorMap(img2, cv2.COLORMAP_VIRIDIS)
    r_img2, g_img2, b_img2 = cv2.split(img2)
    img2 = cv2.merge([b_img2, g_img2, r_img2], 3)

    img3 = copy.deepcopy(roi_area_to_plot)
    r_img3, g_img3, b_img3 = cv2.split(img3)
    # print(r_img3)
    # print(g_img3)
    # print(b_img3)
    #
    # cv2.imwrite(save_path + '/red.png', r_img3)
    # cv2.imwrite(save_path + '/green.png', g_img3)
    # cv2.imwrite(save_path + '/blue.png', b_img3)
    img3 = cv2.merge([b_img3, g_img3, r_img3], 3)
    N_img1, M_img1 = img1.shape[:2]

    cv2.line(img1, pt1=(0, greatest_derivative1), pt2=(N_img1, greatest_derivative1), color=(128, 128, 128),
             thickness=2)
    cv2.line(img2, pt1=(0, greatest_derivative2), pt2=(N_img1, greatest_derivative2), color=(128, 128, 128),
             thickness=2)
    cv2.line(img3, pt1=(0, greatest_derivative3), pt2=(N_img1, greatest_derivative3), color=(128, 128, 128),
             thickness=2)

    fig = plt.figure(figsize=(28, 20))
    fig.add_subplot(331)
    plt.plot(x1, y1, 'r-', label='Wartość centralnego wycinka ROI')
    plt.axvline(greatest_derivative1, 0, 1, label='Punkt narastania do największej zmiany pochodnej')
    plt.ylabel('Znormalizowana średnia wartość kanału maski', fontsize=14)
    plt.xlabel('Wysokość [px]', fontsize=14)
    plt.xticks(np.arange(min(x1), max(x1), 50.0))
    plt.legend(loc='lower left', prop={'size': 12})
    plt.title('Wykryty punkt grubości osadu: ' + str(greatest_derivative1) + ' [px]', fontweight='bold', fontsize=16)
    plt.grid()

    fig.add_subplot(332)
    plt.plot(x2, y2, 'r-', label='Wartość brzegowego wycinka ROI')
    plt.axvline(greatest_derivative2, 0, 1, label='Punkt narastania do największej zmiany pochodnej')
    plt.ylabel('Znormalizowana średnia wartość kanału maski', fontsize=14)
    plt.xlabel('Wysokość [px]', fontsize=14)
    plt.xticks(np.arange(min(x2), max(x2), 50.0))
    plt.legend(loc='upper left', prop={'size': 12})
    plt.title('Wykryty punkt grubości osadu: ' + str(greatest_derivative2) + ' [px]', fontweight='bold', fontsize=16)
    plt.grid()

    fig.add_subplot(333)
    plt.plot(x3, y3, 'r-', label='Wartość iloczynu dwóch obszarów')
    plt.axvline(greatest_derivative3, 0, 1, label='Punkt narastania do największej zmiany pochodnej')
    plt.ylabel('Znormalizowana średnia wartość kanału maski', fontsize=14)
    plt.xlabel('Wysokość [px]', fontsize=14)
    plt.xticks(np.arange(min(x3), max(x3), 50.0))
    plt.legend(loc='upper left', prop={'size': 12})
    plt.title('Wykryty punkt grubości osadu: ' + str(greatest_derivative3) + ' [px]', fontweight='bold', fontsize=16)
    plt.grid()

    fig.add_subplot(334)
    if len(derivatives1) != len(x1):
        del x1[-1]
    plt.plot(x1, derivatives1, 'r--', label='Pierwsza pochodna funkcji S-G centralnego wycinka ROI')
    plt.axvline(max_derivative1, 0, 1, label='Największa pochodna')
    plt.ylabel('Pierwsza pochodna średniej wartości', fontsize=14)
    plt.xlabel('Wysokość [px]', fontsize=14)
    plt.xticks(np.arange(min(x1), max(x1), 50.0))
    plt.legend(loc='lower left', prop={'size': 12})
    plt.title('Punkt największej zmiany pochodnej: ' + str(max_derivative1) + ' [px]', fontweight='bold', fontsize=16)
    plt.grid()

    fig.add_subplot(335)
    if len(derivatives2) != len(x2):
        del x2[-1]
    plt.plot(x2, derivatives2, 'r--', label='Pierwsza pochodna funkcji S-G brzegowego wycinka ROI')
    plt.axvline(max_derivative2, 0, 1, label='Największa pochodna')
    plt.ylabel('Pierwsza pochodna średniej wartości', fontsize=14)
    plt.xlabel('Wysokość [px]', fontsize=14)
    plt.xticks(np.arange(min(x2), max(x2), 50.0))
    plt.legend(loc='upper left', prop={'size': 12})
    plt.title('Punkt największej zmiany pochodnej: ' + str(max_derivative2) + ' [px]', fontweight='bold', fontsize=16)
    plt.grid()

    fig.add_subplot(336)
    if len(derivatives3) != len(x3):
        del x3[-1]
    plt.plot(x3, derivatives3, 'r--', label='Pierwsza pochodna funkcji S-G iloczynu')
    plt.axvline(max_derivative3, 0, 1, label='Największa pochodna')
    plt.ylabel('Pierwsza pochodna średniej wartości', fontsize=14)
    plt.xlabel('Wysokość [px]', fontsize=14)
    plt.xticks(np.arange(min(x3), max(x3), 50.0))
    plt.legend(loc='upper left', prop={'size': 12})
    plt.title('Punkt największej zmiany pochodnej: ' + str(max_derivative3) + ' [px]', fontweight='bold', fontsize=16)
    plt.grid()

    fig.add_subplot(337)
    plt.imshow(img1)
    plt.ylabel('Wysokość [px]', fontsize=14)
    plt.title('Czas:  %s' % time[i], fontweight='bold', fontsize=16)

    fig.add_subplot(338)
    plt.imshow(img2)
    plt.ylabel('Wysokość [px]', fontsize=14)
    plt.title('Czas:  %s' % time[i], fontweight='bold', fontsize=16)

    fig.add_subplot(339)
    plt.imshow(img3)
    plt.ylabel('Wysokość [px]', fontsize=14)
    plt.title('Wykryta grubość osadu:  %s' % thickness + ' [cm]', fontweight='bold', fontsize=16)

    mng = plt.get_current_fig_manager()
    plt.savefig(save_path + '/derivatives%d.png' % i)
    plt.close()


def show(img):
    cv2.imshow('show', img)
    cv2.waitKey(0)


def norm_func(x, *coeffs):
    y = np.polyval(coeffs, x)
    return y


def control_normalization(images, save_path, channel, plot_values):
    image_0 = images[0]

    r = cv2.selectROI(image_0)
    cv2.destroyAllWindows()
    print(r)

    if len(images) < 41:
        time = []
        for i in range(0, len(images)):
            time.append(0)
    elif len(images) == 43:
        time = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 75, 80, 85, 90, 95, 100,
                105, 110, 115, 120, 125, 185, 245, 305, 365, 425, 485, 545, 605, 665, 725,
                785, 845, 905, 965, 1025, 1085, 1145, 1205, 1265]
    else:
        time = list(range(0, 1810, 10))

    images_values = []
    for i in range(0, len(images)):
        images_values.append([])

    for i in range(0, len(images)):
        image = images[i]
        roi_area = image[(r[1]):(r[1]+r[3]), (r[0]):(r[0]+r[2])]
        N, M = roi_area.shape[:2]

        blue_roi_channel, green_roi_channel, red_roi_channel = cv2.split(roi_area)
        if channel == 'R':
            chan = red_roi_channel
        elif channel == 'G':
            chan = green_roi_channel
        elif channel == 'B':
            chan = blue_roi_channel

        n_chan = copy.deepcopy(chan)
        # n_r = copy.deepcopy(r)
        # region_images, central_region = divide_into_regions(n_chan)
        # new_regions_to_draw = regions_to_draw(r, n_r)

        # new_chan = region_images[central_region]
        new_chan = n_chan
        N_new, M_new = new_chan.shape[:2]

        counter = 0
        mask = 1
        images_values[i] = calculate_values(N_new, M_new, mask, counter, plot_values, new_chan, images_values[i])
        # images_values[i] = ndimage.median_filter(images_values[i], size=5)
        # images_values[i] = ndimage.median_filter(images_values[i], size=20)

        # x_tmp = [m for m in range(0, N_new)]
        #
        # fig = plt.figure()
        # plt.plot(x_tmp, images_values[i], 'g', label='Wartość wyliczona przez algorytm')
        # plt.title('Wykres średnich jasności kontroli - wynik pojedycznego obrazu')
        # plt.ylabel('Średnia wartość maski')
        # plt.xlabel('Maska')
        # plt.grid()
        # # plt.legend(loc='lower right', prop={'size': 12})
        # mng = plt.get_current_fig_manager()
        # mng.window.state("zoomed")
        # plt.show()

    x1 = [m for m in range(0, N_new)]

    mean_image_values = []
    for d in range(0, len(x1)):
        tmp_values = []
        for e in range(0, len(images_values)):
            tmp_values.append(images_values[e][d])
        median_value = np.mean(tmp_values)
        mean_image_values.append(median_value)

    mean_image_values = [round(num) for num in mean_image_values]
    y1 = mean_image_values

    max_control_value = max(y1)
    for m in range(0, len(y1)):
        y1[m] = y1[m] / max_control_value
    # y1 = ndimage.median_filter(y1, size=10)
    #
    # fig = plt.figure()
    # plt.plot(x1, y1, 'g', label='Wartość wyliczona przez algorytm')
    # plt.title('Wykres średnich jasności kontroli - przebieg wygładzonych 10 obrazów')
    # plt.ylabel('Średnia wartość maski')
    # plt.xlabel('Maska')
    # plt.grid()
    # # plt.legend(loc='lower right', prop={'size': 12})
    # mng = plt.get_current_fig_manager()
    # mng.window.state("zoomed")
    # plt.show()

    # spl = si.UnivariateSpline(x1, y1, k=2)
    # print(spl)
    # plt.plot(x1, spl(x1), 'r', lw=3)
    # plt.plot(x1, y1, 'go')
    # plt.show()

    f = open("C:/Users/Admin/Desktop/norm.txt", "w")
    fit_results = []
    x = [m for m in range(0, 350)]
    for n in range(2, 6):
        spl = si.UnivariateSpline(x1, y1, k=n)
        result_values = spl(x)
        for i in result_values:
            f.write(str(i) + ", ")
        f.write('\n')
        fit_results.append(result_values)
        print(result_values)
    f.close()

    plt.plot(x1, y1, 'go', label='Dane rzeczywiste')

    for p in range(0, len(fit_results)):
        plt.plot(x1, fit_results[p], alpha=0.6, label=('d = ' + str(p+2)))

    plt.legend(framealpha=1, shadow=True)
    plt.title('Aproksymacja wartości kontrolnych')
    plt.ylabel('Średnia wartość maski')
    plt.xlabel('Maska')
    plt.grid()
    plt.show()

    # DZIAŁAJĄCE SPLINE
    # y_tup = si.splrep(x1, y1, k=3)
    # y_list = list(y_tup)
    # yl = y1
    # # y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]
    # ipl_t = np.linspace(0.0, len(y1) - 1, 400)
    # y_i = si.splev(ipl_t, y_list)
    # print(y_i)

    # fig = plt.figure()
    # plt.plot(ipl_t, e3, label='czas ekspozycji 160 ms')
    # plt.plot(ipl_t, e4, label='czas ekspozycji 80 ms')
    # plt.plot(ipl_t, e5, label='czas ekspozycji 40 ms')
    # plt.plot(ipl_t, e6, label='czas ekspozycji 20 ms')
    # plt.title('Splined y(t)')
    # plt.title('Basis splines')
    # plt.legend(loc='lower center', prop={'size': 12})
    # plt.title('Przebieg wartości kontrolnej', fontweight='bold', fontsize=16)
    # plt.grid()
    # plt.show()

    # DO RYSOWANIA DZIAŁAJĄCYCH SPLINE
    # fig = plt.figure()
    # plt.plot(x1, y1, '-og')
    # plt.plot(ipl_t, y_i, 'r')
    # plt.xlim([0.0, max(x1)])
    # plt.title('Splined y(t)')
    # plt.title('Basis splines')
    # plt.show()

    # poly_fit_sg = scipy.signal.savgol_filter(y1, 9, 2)
    # poly_fit = np.polyfit(x1, poly_fit_sg, 4)
    # poly_fit = np.poly1d(poly_fit)

    # control_model = np.array([-3.86933072e-12, 3.86040748e-09, -1.44804685e-06, 2.59039695e-04,
    # -2.68922792e-02, 2.14580714e+00, 8.57302892e+00])
    # control_model = np.array([-1.08977707e-11, 1.11495683e-08, -4.27845286e-06, 7.69233929e-04,
    # -7.09472456e-02, 3.97233663e+00, 2.52485700e+01])
    # control_model = np.array([1.14697234e-13, -1.52204756e-10, 7.99884235e-08, -2.10853949e-05,
    # 2.91511866e-03, -2.06213603e-01, 7.45805186e+00, 4.17522545e+00])

    # p_values = np.polyval(poly_fit, x1)
    # max_control_value = max(p_values)
    # for m in range(0, len(p_values)):
    #     p_values[m] = p_values[m]/max_control_value

    # TU JEST NORMALIZACJA LS
    # fit_results = []
    # for n in range(3, 10):
    #     p0 = np.ones(n)
    #     popt, pcov = scipy.optimize.curve_fit(norm_func, x1, y1, p0=p0)
    #     fit_results.append(popt)
    #     print(popt)
    #
    # plt.plot(x1, y1, 'go', label='Dane rzeczywiste')
    #
    # xx = np.linspace(min(x1), max(x1), 100)
    # for p in fit_results:
    #     yy = norm_func(xx, *p)
    #     plt.plot(xx, yy, alpha=0.6, label='n = %d' % len(p))
    #
    # plt.legend(framealpha=1, shadow=True)
    # plt.title('Aproksymacja wartości kontrolnych')
    # plt.ylabel('Średnia wartość maski')
    # plt.xlabel('Maska')
    # plt.grid()
    # plt.show()

    # fig = plt.figure(figsize=(28, 20))
    # # plt.plot(x1, poly_fit(x1), 'g--', label='Aproksymacja funkcji (10-go stopnia)')
    # # plt.plot(x1, p_values, 'g--', label='Aproksymowana funkcja kontroli')
    #
    # # plt.plot(x1, poly_fit_sg, 'b-', label='Wartość filtru SG')
    #
    # plt.plot(x1, y2, label='Fitted function')
    # plt.plot(x1, y1, 'r-', label='Wartość kontrolna średniej')
    # plt.ylabel('Średnia wartość odcinka', fontsize=14)
    # plt.xlabel('Wysokość [px]', fontsize=14)
    # plt.xticks(np.arange(min(x1), max(x1), 50.0))
    # plt.legend(loc='lower left', prop={'size': 12})
    # plt.title('Przebieg wartości kontrolnej', fontweight='bold', fontsize=16)
    # plt.grid()
    # mng = plt.get_current_fig_manager()
    # mng.window.state("zoomed")
    # plt.savefig(save_path + '/control_profile.png')
    # plt.close()


def average_from_median_outputs(median, mask):
    average_output = []
    first_id = 0
    for i in range(0, len(median)-mask+1):
        average_mask = []
        for j in range(0, mask):
            average_mask.append(median[first_id + j])
        average_output.append(np.mean(average_mask))
        first_id = first_id + 1

    average_time = []
    for i in range(0, len(average_output)):
        average_time.append(10 + i)

    return average_output, average_time


def sludge_thickness_median(save_path, median_filter_size, smoothing, tilt,
                            calibrate, to_normalize, given_height, median_results):

    t = Tk()
    t.withdraw()
    folder = filedialog.askdirectory(initialdir=os.getcwd(), title='Wybierz folder.')
    images, loaded_folder = load_images_from_folder(folder)

    window_analysis = "mean"
    save_name = "default"
    if tilt == "yes":
        tile_angle = rotation_angle(images)
        if calibrate == "yes":
            image_0 = images[0]
            image_0 = camera_calibration(image_0)
            image_0 = image_rotation(image_0, tile_angle)
    elif tilt == "no":
        image_0 = images[0]
        if calibrate == "yes":
            image_0 = images[0]
            image_0 = camera_calibration(image_0)

    r = cv2.selectROI(image_0)
    cv2.destroyAllWindows()
    print(r)

    processing_mask_size = median_results  # default 10
    first_image_id = 0
    time = list(range(0, len(images)-processing_mask_size+1))

    images_values = []
    images_values_2 = []
    images_values_product = []
    number_of_images = int(len(images)-processing_mask_size+1)
    for i in range(0, number_of_images):
        images_values.append([])
        images_values_2.append([])
        images_values_product.append([])

    workbook = xlsxwriter.Workbook(save_path + '/Data.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    times_to_plot = []
    thickness = []

    F = open(save_path + '/list.txt', 'w')

    for i in range(0, len(images)-processing_mask_size+1):
        tmp_val1 = []
        tmp_val2 = []
        tmp_val3 = []
        tmp_x1 = []
        tmp_x2 = []
        tmp_x3 = []
        tmp_thickness = []
        imgs1 = []
        imgs2 = []
        imgs3 = []
        tmp_derivatives1 = []
        tmp_derivatives2 = []
        tmp_derivatives3 = []
        tmp_new_greatest_derivative1 = []
        tmp_new_greatest_derivative2 = []
        tmp_new_greatest_derivative3 = []
        tmp_greatest_derivative1 = []
        tmp_greatest_derivative2 = []
        tmp_greatest_derivative3 = []

        group = []

        for j in range(0, processing_mask_size):
            group.append(images[first_image_id + j])
            image = group[j]
            if calibrate == "yes":
                image = camera_calibration(image)
                bl, gr, re = cv2.split(image)
                image = cv2.merge([re, gr, bl], 3)

            if tilt == "yes":
                image = image_rotation(image, tile_angle)
            roi_area = image[(r[1]):(r[1] + r[3]), (r[0]):(r[0] + r[2])]
            N, M = roi_area.shape[:2]

            blue_roi_channel, green_roi_channel, red_roi_channel = cv2.split(roi_area)

            chan = red_roi_channel

            n_chan = copy.deepcopy(chan)
            n_r = copy.deepcopy(r)
            region_images, central_region = divide_into_regions(n_chan)
            new_regions_to_draw = regions_to_draw(r, n_r)

            new_chan = region_images[central_region]
            N_new, M_new = new_chan.shape[:2]

            counter = 0
            mask = 1
            tmp_v1 = []
            tmp_v1 = calculate_values(N_new, M_new, mask, counter, window_analysis, new_chan, tmp_v1)

            if to_normalize == "yes":
                max_value1 = max(tmp_v1)
                for l1 in range(0, len(tmp_v1)):
                    tmp_v1[l1] = tmp_v1[l1] / max_value1
                tmp_v1, control_values1 = normalization(tmp_v1, N_new)

            if smoothing == "yes":
                tmp_v1 = ndimage.median_filter(tmp_v1, size=median_filter_size)

            n_image = copy.deepcopy(image)
            second_roi_width = 15
            roi_area_to_plot = n_image[(r[1]):(r[1] + r[3]), (r[0]):(r[0] + r[2])]
            N, M = roi_area.shape[:2]
            roi_area = roi_area_to_plot[0:N, 0:second_roi_width]
            N, M = roi_area.shape[:2]

            blue_roi_channel, green_roi_channel, red_roi_channel = cv2.split(roi_area)
            chan = red_roi_channel

            tmp_v2 = []
            tmp_v2 = calculate_values(N, M, mask, counter, window_analysis, chan, tmp_v2)

            if to_normalize == "yes":
                tmp_v2, control_values2 = normalization(tmp_v2, N)

            if smoothing == "yes":
                tmp_v2 = ndimage.median_filter(tmp_v2, size=median_filter_size)

            if to_normalize == "no":
                max_value1 = max(tmp_v1)
                for l1 in range(0, len(tmp_v1)):
                    tmp_v1[l1] = tmp_v1[l1] / max_value1

            max_value2 = max(tmp_v2)
            for l2 in range(0, len(tmp_v2)):
                tmp_v2[l2] = tmp_v2[l2] / max_value2

            for c1 in range(0, len(tmp_v2)):
                tmp_v2[c1] = 1 - tmp_v2[c1]

            tmp_v3 = []
            for c in range(0, len(tmp_v2)):
                tmp_v3.append(tmp_v1[c] * tmp_v2[c])

            y1 = tmp_v1
            x1 = [m for m in range(0, N_new)]
            delta1 = x1[1] - x1[0]
            derivatives1 = np.diff(y1) / delta1

            tmp_x1.append(x1)
            greatest_derivative1 = np.where(derivatives1 == max(derivatives1, key=abs))

            y2 = tmp_v2
            x2 = [n for n in range(0, N)]
            delta2 = x2[1] - x2[0]
            derivatives2 = np.diff(y2) / delta2

            tmp_x2.append(x2)
            greatest_derivative2 = np.where(derivatives2 == max(derivatives2, key=abs))

            y3 = tmp_v3
            x3 = [n for n in range(0, N)]
            delta3 = x3[1] - x3[0]
            derivatives3 = np.diff(y3) / delta3

            tmp_x3.append(x3)
            greatest_derivative3 = np.where(derivatives3 == max(derivatives3, key=abs))

            tmp_val1.append(y1)
            tmp_val2.append(y2)
            tmp_val3.append(y3)

            tmp_derivatives1.append(derivatives1)
            tmp_derivatives2.append(derivatives2)
            tmp_derivatives3.append(derivatives3)

            tmp_greatest_derivative1.append(greatest_derivative1)
            tmp_greatest_derivative2.append(greatest_derivative2)
            tmp_greatest_derivative3.append(greatest_derivative3)

            img1 = copy.deepcopy(new_chan)
            img2 = copy.deepcopy(roi_area)
            r_img2, g_img2, b_img2 = cv2.split(img2)
            img2 = r_img2
            if to_normalize == "no":
                img2 = 255 - img2
            img3 = copy.deepcopy(roi_area_to_plot)

            new_greatest_derivative1 = 0
            tmp = greatest_derivative1[0][0]
            for d in range((greatest_derivative1[0][0] - 1), 0, -1):
                if abs(derivatives1[d]) < tmp:
                    tmp = abs(derivatives1[d])
                    new_greatest_derivative1 = d
                else:
                    break

            new_greatest_derivative2 = 0
            tmp = greatest_derivative2[0][0]
            for d in range((greatest_derivative2[0][0] - 1), 0, -1):
                if abs(derivatives2[d]) < tmp:
                    tmp = abs(derivatives2[d])
                    new_greatest_derivative2 = d
                else:
                    break

            new_greatest_derivative3 = 0
            tmp = greatest_derivative3[0][0]
            for d in range((greatest_derivative3[0][0] - 1), 0, -1):
                if abs(derivatives3[d]) < tmp:
                    tmp = abs(derivatives3[d])
                    new_greatest_derivative3 = d
                else:
                    break

            tmp_new_greatest_derivative1.append(new_greatest_derivative1)
            tmp_new_greatest_derivative2.append(new_greatest_derivative2)
            tmp_new_greatest_derivative3.append(new_greatest_derivative3)

            if to_normalize == "yes":
                tmp_img1 = copy.deepcopy(img1)
                N_tmp1, M_tmp1 = tmp_img1.shape[:2]
                for k in range(0, N_tmp1):
                    for k2 in range(0, M_tmp1):
                        tmp_img1[k][k2] = tmp_img1[k][k2] / control_values1[k]
                img1 = tmp_img1

                tmp_img2 = copy.deepcopy(img2)
                N_tmp2, M_tmp2 = tmp_img2.shape[:2]
                for k in range(0, N_tmp2):
                    for k2 in range(0, M_tmp2):
                        tmp_img2[k][k2] = tmp_img2[k][k2] / control_values2[k]
                img2 = tmp_img2
                img2 = 255 - img2

            imgs1.append(img1)
            imgs2.append(img2)
            imgs3.append(img3)

            N_img3, M_img3, D_img3 = img3.shape
            real_height = given_height
            detected_height = round(((N_img3 - new_greatest_derivative3) * real_height) / N_img3, 2)
            tmp_thickness.append(detected_height)

        first_image_id = first_image_id + 1
        median_thickness = np.where(tmp_thickness == np.array(statistics.median_high(tmp_thickness)))
        median_thickness = median_thickness[0][0]

        img1 = imgs1[median_thickness]
        img2 = imgs2[median_thickness]
        img3 = imgs3[median_thickness]
        derivatives1 = tmp_derivatives1[median_thickness]
        derivatives2 = tmp_derivatives2[median_thickness]
        derivatives3 = tmp_derivatives3[median_thickness]
        new_greatest_derivative1 = tmp_new_greatest_derivative1[median_thickness]
        new_greatest_derivative2 = tmp_new_greatest_derivative2[median_thickness]
        new_greatest_derivative3 = tmp_new_greatest_derivative3[median_thickness]
        greatest_derivative1 = tmp_greatest_derivative1[median_thickness]
        greatest_derivative2 = tmp_greatest_derivative2[median_thickness]
        greatest_derivative3 = tmp_greatest_derivative3[median_thickness]
        y1 = tmp_val1[median_thickness]
        y2 = tmp_val2[median_thickness]
        y3 = tmp_val3[median_thickness]
        x1 = tmp_x1[median_thickness]
        x2 = tmp_x2[median_thickness]
        x3 = tmp_x3[median_thickness]

        detected_height = tmp_thickness[median_thickness]
        thickness.append(detected_height)

        plot_derivatives(img1, img2, img3, derivatives1, derivatives2, derivatives3,
                         new_greatest_derivative1, new_greatest_derivative2, new_greatest_derivative3,
                         greatest_derivative1[0][0], greatest_derivative2[0][0], greatest_derivative3[0][0], x1, x2, x3,
                         y1, y2, y3, time, i, detected_height, save_path)

        print('Gotowy czas ' + str(time[i]))

        worksheet.write(row, col, 'Czas: ' + str(time[i]) + 's')
        worksheet.write(row, col + 1, 'Grubość: ' + str(detected_height) + ' cm')
        worksheet.write(row, col + 2, 'Punkt zmiany pochodnej: ' + str(greatest_derivative3[0][0]))
        tmp_col = col
        row = row + 1
        worksheet.write(row, tmp_col, 'Maska:')
        tmp_col = tmp_col + 1
        #
        for value in x2:
            worksheet.write(row, tmp_col, value)
            tmp_col = tmp_col + 1

        tmp_col = col
        row = row + 1
        worksheet.write(row, tmp_col, 'Iloczyn:')
        tmp_col = tmp_col + 1

        for value in y3:
            worksheet.write(row, tmp_col, value)
            tmp_col = tmp_col + 1

        row = row + 2
    workbook.close()

    predicted_value = 15
    started_len = len(thickness)

    to_delete = [i for i, v in enumerate(thickness) if v >= predicted_value]

    new_time = []
    val = 10
    for k in range(0, len(thickness)):
        new_time.append(val)
        val = val + 1

    for ele in range(len(new_time) - 1, -1, -1):
        if ele in to_delete:
            del to_delete[-1]
            del new_time[ele]
            del thickness[ele]

    ended_len = len(thickness)
    number_of_deleted_values = started_len - ended_len

    fig = plt.figure()
    plt.plot(new_time, thickness, 'go')
    plt.title('Wykres sedymentacji osadu w czasie. Usunięte wartości odstające: ' + str(number_of_deleted_values))
    plt.ylabel('Odczytana wartość grubości osadu [cm]')
    plt.xlabel('Czas [s]')
    plt.grid()
    mng = plt.get_current_fig_manager()
    mng.window.state("zoomed")
    plt.savefig(save_path + '/' + save_name + '.png')
    plt.close()

    for i in thickness:
        F.write(str(i) + ", ")

    F.close()


def main():
    save_path = 'C:/Users/Admin/Desktop'
    sludge_thickness_median(save_path, 7, "yes", "no", "no", "no", 15.6, 1)
    # 1. Ścieżka do wczytanych obrazów - wywołana po uruchomieniu funkcji
    # 2. Ścieżka do zapisu
    # 3. szerokość okna filtru medianowego
    # 4. Zastosować filtr?
    # 5. Korekcja na przekrzywienie?
    # 6. Korekcja na zniekształcenie soczewki?
    # 7. Normalizacja?
    # 8. Rzeczywista wysokość
    # 9. Liczba obrazów do wyliczenia mediany


main()
