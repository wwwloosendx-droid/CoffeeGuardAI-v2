// CoffeeGuard AI — dashboard interactivity
document.addEventListener("DOMContentLoaded", function () {

  /* ---------- Disease distribution pie chart ---------- */
  const pieCanvas = document.getElementById("diseaseChart");
  if (pieCanvas && window.DASHBOARD_DATA) {
    const { chartLabels, chartValues } = window.DASHBOARD_DATA;
    new Chart(pieCanvas, {
      type: "doughnut",
      data: {
        labels: chartLabels,
        datasets: [{
          data: chartValues,
          backgroundColor: ["#2f7d4f", "#d9912b", "#b83f35", "#3d6b99", "#6b4423", "#4ea674"],
          borderWidth: 2,
          borderColor: "#ffffff"
        }]
      },
      options: {
        plugins: { legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } } },
        cutout: "62%"
      }
    });
  }

  /* ---------- Confidence / yield trend line chart ---------- */
  const trendCanvas = document.getElementById("trendChart");
  if (trendCanvas && window.DASHBOARD_DATA) {
    const { trendLabels, confidenceTrend, yieldTrend } = window.DASHBOARD_DATA;
    new Chart(trendCanvas, {
      type: "line",
      data: {
        labels: trendLabels,
        datasets: [
          {
            label: "Confidence %",
            data: confidenceTrend,
            borderColor: "#3d6b99",
            backgroundColor: "rgba(61,107,153,0.08)",
            tension: 0.35,
            fill: true,
            pointRadius: 3
          },
          {
            label: "Yield Outlook %",
            data: yieldTrend,
            borderColor: "#2f7d4f",
            backgroundColor: "rgba(47,125,79,0.08)",
            tension: 0.35,
            fill: true,
            pointRadius: 3
          }
        ]
      },
      options: {
        scales: { y: { beginAtZero: true, max: 100 } },
        plugins: { legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } } }
      }
    });
  }

  /* ---------- Live clock in topbar ---------- */
  const clockEl = document.getElementById("liveClock");
  if (clockEl) {
    const updateClock = () => {
      const now = new Date();
      clockEl.textContent = now.toLocaleString([], {
        weekday: "short", year: "numeric", month: "short",
        day: "numeric", hour: "2-digit", minute: "2-digit"
      });
    };
    updateClock();
    setInterval(updateClock, 30000);
  }

  /* ---------- Export history to CSV ---------- */
  const exportBtn = document.getElementById("exportCsvBtn");
  if (exportBtn && window.DASHBOARD_DATA && window.DASHBOARD_DATA.historyRows) {
    exportBtn.addEventListener("click", function () {
      const rows = window.DASHBOARD_DATA.historyRows;
      let csv = "Date,Result,Confidence(%),Yield Outlook(%)\n";
      rows.forEach(r => {
        csv += `"${r.date}","${r.result}",${r.confidence},${r.yieldEstimate}\n`;
      });
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "coffeeguard_history.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    });
  }
});
