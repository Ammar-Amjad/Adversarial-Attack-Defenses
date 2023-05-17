# Adversarial Attack Defenses in Machine Learning

This repository contains a comprehensive research on various defense techniques against adversarial attacks and Membership Inference Attacks (MIA) on machine learning models. The study utilizes the Fashion-MNIST dataset and a convolutional neural network (CNN) model to evaluate the effectiveness of each defense technique. 

The [video presentation](https://drive.google.com/file/d/1tVLzJYMn-DYwJ1E2j3f7-iJgjmv2-NED/view?usp=share_link) can be found on drive.

## Key Defense Techniques Studied
1. Certified Robustness via Randomized Smoothing
2. Runtime Adversarial Training
3. Input Preprocessing (Filtering Techniques)

## Results
While all the defense strategies showed promise, there were trade-offs between model performance and robustness against adversarial examples. The most effective defense was found to be Median Filtering with a dual filter technique.
Detailed experimental results and anlysis can be found in the [Report](https://github.com/Ammar-Amjad/Adversarial-Attack-Defenses/blob/main/Report.pdf).

## Future Research
Future research will focus on analyzing ensemble models and certified robustness techniques, and extending investigations to tabular and text-based datasets.

## How to Run
1. Clone the repository.
2. Run the model training and testing script.
   ```
   python model.py
   ```

## Dependencies
- Python 3.6+
- TensorFlow 2.5+
- Numpy 1.19+
- Scikit-learn 0.24+
- Matplotlib 3.3+

## Contributing
Please feel free to raise issues or submit pull requests.

## Citation
If you find our research helpful, please cite our work.
