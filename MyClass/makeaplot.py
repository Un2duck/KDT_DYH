# ---------------------------------------------------------------------
# Version.1
# file_name : makeaplot.py
# Date : 2024-09-18
# 설명 : 각종 그래프 함수
# ---------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib

# ---------------------------------------------------------------------
# 함수 이름 : make_bar
# 매개 변수 : counts, target_key='key', changed_color='darkred', default_color='darksalmon', name='name'
# 함수 역할 : bar plot 그리기, 색깔 2가지
# ---------------------------------------------------------------------

def make_bar(counts, target_key='key', changed_color='darkred', default_color='darksalmon', name='name'):
    colors = [changed_color if key in target_key else default_color for key in counts.keys()]
    plt.title(f'[횟수]', size=35)
    a = plt.bar(counts.keys(), counts.values(), color = colors)
    plt.bar_label(a, label_type='edge')
    plt.xlabel(name, size=25)
    plt.ylabel('빈도', size=25)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# 함수 이름 : make_heatmap
# 매개 변수 : DF, cmapcolor='Greens' (default)
# 함수 역할 : DF에 대하여 heatmap 시각화 진행
# ---------------------------------------------------------------------

def make_heatmap(DF, cmapcolor='Greens'):
    plt.figure(figsize=(10,7))
    sns.heatmap(DF.corr(), annot=True, fmt=".2f", cmap=cmapcolor)
    plt.show()

# ---------------------------------------------------------------------
# 함수 이름 : loss_score_plot
# 매개 변수 : loss, score, threshold=10 (default)
# 함수 역할 : 학습 후 loss, score 시각화 진행
# ---------------------------------------------------------------------

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