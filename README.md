# Adversarial Attack Defenses in Machine Learning

This repository contains a comprehensive research on various defense techniques against adversarial attacks and Membership Inference Attacks (MIA) on machine learning models. The study utilizes the Fashion-MNIST dataset and a convolutional neural network (CNN) model to evaluate the effectiveness of each defense technique. 

The video presentation can be found on [youtube](https://drive.google.com/file/d/1tVLzJYMn-DYwJ1E2j3f7-iJgjmv2-NED/view?usp=share_link).

## Key Defense Techniques Studied
1. Certified Robustness via Randomized Smoothing
2. Runtime Adversarial Training
3. Input Preprocessing (Filtering Techniques)

## Contents
- **certified_robustness.py**: Implementation of Certified Robustness via Randomized Smoothing.
- **runtime_adversarial_training.py**: Implementation of Runtime Adversarial Training using FGSM.
- **input_preprocessing.py**: Implementation of various Input Preprocessing or Filtering Techniques.
- **model.py**: The CNN model used for the study, including training and testing scripts.
- **evaluation.py**: Scripts for evaluating the performance of each defense technique.

## Results
While all the defense strategies showed promise, there were trade-offs between model performance and robustness against adversarial examples. The most effective defense was found to be Median Filtering with a dual filter technique.

## Future Research
Future research will focus on analyzing ensemble models and certified robustness techniques, and extending investigations to tabular and text-based datasets.

## How to Run
1. Clone the repository.
2. Run the script for each defense technique.
   ```
   python certified_robustness.py
   python runtime_adversarial_training.py
   python input_preprocessing.py
   ```
3. Run the model training and testing script.
   ```
   python model.py
   ```
4. Run the evaluation script.
   ```
   python evaluation.py
   ```

## Dependencies
- Python 3.6+
- TensorFlow 2.5+
- Numpy 1.19+
- Scikit-learn 0.24+
- Matplotlib 3.3+

## Contributing
We welcome contributions to this project. Please feel free to raise issues or submit pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you find our research helpful, please cite our work.
