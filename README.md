# SoccerNet Challenge: Jersey Number Recognition
This project works on tackling SoccerNet Jersey Number Recognition Task using Deep Learning techniques. We aim to make a football jersey number recognition system that can automatically detect and identify the number on the athlete's jersey. If the number is not detected, it will output $-1$; if the number is detected, it will output the corresponding number.

We improve the Baseline Model's inference time by first performing pose detection and then using semi supervised learning using EasyOCR to train a light weight classfier model.

![Model Architecture Updated](https://github.com/user-attachments/assets/0e6f11c9-5a77-487b-a9bf-cedcc01baebc)


## Inference Setup
Create a new virtual env using
```sh
python -m venv .venv
```
and then activate it with
```sh
source .venv/bin/activate
```
Now, install the requirements using
```sh
pip install -r requirements.txt
```
Finally, you can run inference by executing
```sh
python inference.py
```

### Downloading Dataset
You can download `SoccerNet` dataset using the `download_data.py` script. To download the dataset used for training the classifier model, refer to the [Google Drive](https://drive.google.com/drive/folders/1qW14QyPeIMMp3Z5wLWgTD9ijMFaUEfIn?usp=sharing) link and use `soccernet-image-legibility-classifier-training.ipynb` for training the classifier model.

### Pre trained Weights
Classifier weights can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1qW14QyPeIMMp3Z5wLWgTD9ijMFaUEfIn?usp=sharing). Place these under `models/` directory.

### Live Demo
To run a live version of the model, execute
```sh
python demo.py
```

## Results
We compare our model against the Baseline [A General Framework for Jersey Number Recognition in Sports](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf) (Maria Koshkina, James H. Elder).


### Table 1: Accuracy Comparison (%)
| Method                     | Challenge Set | Test Set |
|----------------------------|--------------|----------|
| Baseline (Koshkina et al.) | 79.31 %         | 87.45 %     |
| Ours                       | 72.79 %         | 80.18 %     |

### Table 2: Inference Time Comparison (sec/tracklet)
| Method                     | Total Inference Time | Pose Detection | Classifier | STR  |
|----------------------------|----------------------|----------------|------------|------|
| Baseline (Koshkina et al.) | 16.87                 | 4.41           | 10.84       | 1.62 |
| Ours                       | **10.42**                 | 8.17           | **0.17**       | 2.08 |

The above is based on running the models on 2 T4 GPUs on Test set of SoccerNet.

For our classifier model, we created a new image-level dataset from EasyOCR and ResNet34 trained on Hockey dataset and did a 10-90 (10% Train and 90% Test) split on the dataset. We got the following accuracy

### Table 3: Legiblity Classifier
| Train                     | Test |
|----------------------------|--------------|
| 99.03 % | 98.31 %      |


Our Model performs at $72.79~\%$ on the Challenge Dataset as can be seen on our EvalAI submission below

### Table 4: EvalAI Submission
| Team Name    | Method Name | Status   | Execution Time (sec.) | Submitted File | Result File | Stdout File | Stderr File | Submitted At |
|-------------|------------|----------|----------------------|----------------|-------------|-------------|-------------|--------------------------|
| UBC 419 T2  | UBC Team 2 | finished | 0.169408             | [Link](https://evalai.s3.amazonaws.com/media/submission_files/submission_502585/bcad9e8f-9baa-4b73-81c0-64a5aa60d0ab.json) | [Link](https://evalai.s3.amazonaws.com/media/submission_files/submission_502585/bcd13e13-a8db-4565-ba77-876ff78d7c2a.json) | [Link](https://evalai.s3.amazonaws.com/media/submission_files/submission_502585/b17a55ad-8d6e-435e-80ba-84f2af518f6c.txt) |  | 2025-03-29 22:52:42.173476+00:00 |
