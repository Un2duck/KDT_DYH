# ---------------------------------------------------------------------
# Version.1
# file_name : OpenCV_func.py
# Date : 2024-09-28
# 설명 : OpenCV Project용 함수
# ---------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib

def loss_score_plot(loss, score, threshold=10):
    fg, axes=plt.subplots(1,2,figsize=(10,5))
    axes[0].plot(range(1, threshold+1), loss[0][:threshold], label='Train')
    axes[0].plot(range(1, threshold+1), loss[1][:threshold], label='Val')
    axes[0].grid()
    axes[0].legend()
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Epoch&Loss')

    axes[1].plot(range(1, threshold+1), score[0][:threshold], label='Train')
    axes[1].plot(range(1, threshold+1), score[1][:threshold], label='Val')
    axes[1].grid()
    axes[1].legend()
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Epoch&Score')
    plt.tight_layout()
    plt.show()