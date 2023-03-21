The calculation of image similarity of comprehensive two-dimensional gas chromatographic ( GC×GC ) fingerprints
==

SURF-FLANN algorithm is a feature detection-matching algorithm. Feature detection is a concept in the field of computer vision and image processing. It means that it extracts image information through the computer and decides whether each point belongs to an image feature. By extracting, describing and matching feature points of the image, the detection and matching of the feature points in two GC×GC fingerprints are completed. The main steps include: (1) SURF feature extraction; (2) feature descriptor generation; (3) FLANN feature matching; (4) similarity computation. In other words, the similarity of GC×GC fingerprints is calculated according to the number of matching feature points.
Image complexity usually keep the similarity scoring ambiguity in non-targeted GC×GC fingerprints, including those data from complex essential oil or atomosphere samples. Therefore, targeted filtration should be introduced in pre-processing procedure, i.e.: retaining target features for fingerprint evaluation and removing interfering features. In our paer, zone- specific ion filtration was introduced for quality evaluation of GC×GC fingerprints. 
We visualize the SURF-FLANN algorithm as a tool for readers to use. The tool is shown below:
 
Requirements
==
•	Python, version 3.7 or earlier

•	OpenCV 3.4.2

•	Windows 11

•	Install additional libraries, listed in the requirements.txt

Usage
==
•	Prepare two GC×GC fingerprints as the queryImage and referenceImage
  (raw data; targeted filtration; zone- specific ion filtration; etc)
  
•	Run SF.py, input the queryImage and referenceImage, then get the matching result and get the similarity value showed on the interface

