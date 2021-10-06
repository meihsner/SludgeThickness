# SludgeThickness
The main goal of the master's thesis was to develop a new algorithm using the vision method and laser beam illumination,
allowing to estimate the thickness of the sediment in the reservoir to a greater extent, without interfering with the studied environment.
The assumptions of the work included:
- selection of the optimal method of measuring the thickness of the sediment in the tank,
- selection of parameters and time points for image acquisition,
- verification of measurements on the basis of a manually determined amount of sediment in the tank, examining the dependence of the amount of mass on the measured value,
- examining the dependence of the sludge thickness on time (sludge sedimentation profile),
- examining various types of research materials,
- examining and eliminating the influence of the research environment on the measurement,
- examination and elimination of the influence of imperfections of the vision system,
- examining the influence of data filtration on the obtained results.

# Libraries:
- *Tkinter* - manual selection of the path to the images (GUI) (8.6.0),
- *OpenCV* - determining the region of interest, performing normalization, correcting lens distortion, assisting in data visualization and image acquisition. (4.5.1.48),
- *Pillow* - reading the image as an object (8.2.0),
- *XlsxWriter* - collecting the results of the sediment thickness measurement over time and saving to an Excel spreadsheet (1.3.9),
- *matplotlib* - presentation of normalized images on the basis of a control sample, visualization of the course of the first derivative and
plotting the sedimentation profile (3.3.1),
- *NumPy* - calculation of average pixel values, polynomial regression, or searching for minimum, maximum or significant values for the analysis (1.20.2),
- *SciPy* - ﬁltration of data with a Savitzky-Golay filter, median filter and to obtain the best polynomial coefficients of the control sample
- other built-in libraries, such as: os, math, copy

# Workplace and research material:
## Material:
- wheat flour,
- sand taken from the external environment.
## Research position:
- Creative Live Cam Chat HD camera placed on a tripod,
- BOSCH PLL5 laser level,
- a vessel with an octagonal base (21 cm high, 8.5 cm in diameter).

# Sample results of the developed algorithm:
The value calculated by the algorithm is marked in green, and the manually determined value in red.

<p align="center">
  <img src="https://user-images.githubusercontent.com/91888660/136269405-c3ebadd5-6ede-420f-a7f2-1325806c5dca.png" width="800">
</p>
Picture 1. Flour sedimentation chart with a manual verification sample (22g)

<p align="center">
  <img src="https://user-images.githubusercontent.com/91888660/136269459-2369671e-dd17-4756-88b4-c92d06be2589.png" width="800">
</p>
Picture 2. Flour sedimentation chart with a manual verification sample (66g)

<p align="center">
  <img src="https://user-images.githubusercontent.com/91888660/136269530-774fd5f6-8ca5-4a3e-8fd5-cb7499e7af09.png" width="800">
</p>
Picture 3. Sand sedimentation chart with a manual verification sample

<p align="center">
  <img src="https://user-images.githubusercontent.com/91888660/136269604-cd6cfd38-f1ba-48da-a793-9a8a7ab90d76.png" width="800">
</p>
Picture 4. Presentation of a single point in time and intermediate stages of the analysis
