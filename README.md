# Intelligent Tutoring System recognizing Comprehension Problems during Learning
This project is meant to develop a (low-cost) intelligent tutoring system monitoring the learning and comprehension 
process of complex material during e.g. online classes or e-learning environments. The implementation is designed to use
 the system with a notebook without the need for a GPU or other resources, such as Clouds, to comply with data protection.
In this case, the intelligent tutoring system was tested on a language grammar learning task. A novel stimulus was 
presented to learn the basic Finnish plural in a short video. The learner must take a comprehension test directly after 
the explanation video unless he or she showed signs of comprehension difficulties. in case of comprehension difficulties,
 the tutorial system suggests a new explanation in which the plural rules are shown again in summary.
 
This repository contains only the tutoring system. You can find the repository for the model in my [Github](https://github.com/maskaljunas/ComprehensionProblems_DAiSEE). 
The model was developed using the DAiSEE Dataset containing multi-label videos capturing the engagement and emotional states
(frustration, confusion and boredom) of students watching MOOCs. Based on the cultural origin of the students in the 
dataset and the classification accuracy, the model is very sensitive to signs of comprehension problems, as used in this 
project. A better model is in development, and I am open to improvements from others.


![](fig/Architecture.PNG)

# Quick Start
Make sure that OpenFace is installed correctly according to your operating system. E.g. for Windows see the [Wiki](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation).

        $ pip install -r requirements.txt
        
        
 Run the tutoring system:
 
        $ main.py


A detailed description of the implementation and how to adapt the system to your hardware will follow.

