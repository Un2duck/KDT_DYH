function createPopupChart(labels, values1, values2, values3, elecErrorMargin) {
    const ctx = document.getElementById('popup-chart').getContext('2d');

    if (window.popupChart) {
        window.popupChart.destroy();
    }

    const upperBound = values3.map((value, index) => value + elecErrorMargin[index]);
    const lowerBound = values3.map((value, index) => value - elecErrorMargin[index]);

    window.popupChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: '수도 사용량',
                    data: values1,
                    backgroundColor: 'rgba(53, 183, 255, 0.4)',
                    borderColor: '#35B7E0',
                    borderWidth: 3,
                    tension: 0.3,
                },
                {
                    label: '전기 사용량',
                    data: values2,
                    backgroundColor: 'rgba(229, 163, 6, 0.4)',
                    borderColor: '#E5A306',
                    borderWidth: 3,
                    tension: 0.3,
                },
                {
                    label: '전기 예측',
                    data: values3,
                    borderColor: '#E8D400',
                    borderWidth: 3,
                    borderDash: [5, 5],
                },
                {
                    label: '오차 범위 상한',
                    data: upperBound,
                    borderColor: 'rgba(232, 212, 0, 0)',
                    backgroundColor: 'rgba(232, 212, 0, 0.2)',
                    fill: '+1',
                },
                {
                    label: '오차 범위 하한',
                    data: lowerBound,
                    borderColor: 'rgba(232, 212, 0, 0)',
                    backgroundColor: 'rgba(232, 212, 0, 0.2)',
                    fill: '-1',
                },
            ],
        },
    });
}
