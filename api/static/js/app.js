import {
    getAvailabilityColor,
    getPredictionColor,
    createSingleMarker,
    createDualMarker,
    parseStations,
    formatPredictionTime
} from './helpers.js';
  
  let map;
  let stationMarkers = [];
  
  function clearMarkers() {
    stationMarkers.forEach(marker => map.removeLayer(marker));
    stationMarkers = [];
  }
  
  function addStationMarkers(stations, predictions = null) {
    clearMarkers();
  
    stations.forEach(station => {
      const predictionFromArray = predictions?.find(p => p.station_id === station.station_id);
      const predictedAvailability = predictionFromArray?.predicted_availability || station.predicted_availability;
      const predictionTime = predictionFromArray?.prediction_time || station.prediction_time;
      const horizonHours = predictionFromArray?.horizon_hours ?? station.horizon_hours;
      const currentColor = getAvailabilityColor(station.num_bikes_available);
      const predictionColor = predictedAvailability ? getPredictionColor(predictedAvailability) : null;
  
      const marker = predictionColor
        ? createDualMarker(station, currentColor, predictionColor)
        : createSingleMarker(station, currentColor);
  
      marker.bindPopup(`
        <div class="station-popup">
          <strong>${station.name}</strong><br>
          <div class="bike-count">🚲 ${station.num_bikes_available} bikes</div>
          ${predictedAvailability ? `<div>🔮 In ${horizonHours}h (at ${formatPredictionTime(predictionTime, horizonHours)}): <strong>${predictedAvailability}</strong></div>` : ''}
          <div class="dock-count">🅿️ ${station.num_docks_available} docks</div>
        </div>
      `);
  
      marker.addTo(map);
      stationMarkers.push(marker);
    });
  
    const el = document.getElementById('last-updated');
    if (el) el.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
  }
  
  function initMap() {
    map = L.map('map').setView([41.8781, -87.6298], 12);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors'
    }).addTo(map);
  }
  

  function updateAutoRefreshMode() {
    const predictionModeInput = document.querySelector('input[name="view-mode"]:checked');
    const mode = predictionModeInput?.value || 'current';
    const stationDataDiv = document.getElementById('station-data');
    
    if (stationDataDiv) {
      // Update the auto-refresh URL to include the current mode
      stationDataDiv.setAttribute('hx-get', `/api/stations?mode=${mode}`);
    }
  }

  function setupHTMX() {
    // Listen for radio button changes to update auto-refresh mode
    document.body.addEventListener('change', function(event) {
      if (event.target.name === 'view-mode') {
        updateAutoRefreshMode();
      }
    });

    document.body.addEventListener('htmx:afterSwap', function(event) {
        if (event.detail.target.id !== 'station-data') return;

        const stations = parseStations(event.detail.target);
        if (!stations.length) return;

        // Check if we're in prediction mode
        const predictionModeInput = document.querySelector('input[name="view-mode"]:checked');
        const predictionMode = predictionModeInput?.value === 'predicted';
        const statusDiv = document.getElementById('prediction-status');

        if (predictionMode && statusDiv) {
          const hasPredictions = stations.some(station => Boolean(station.predicted_availability));

          if (!hasPredictions) {
              statusDiv.textContent = '⚠️ ML service unavailable - showing current data';
              return;
          }
      } else if (statusDiv) {
          statusDiv.textContent = '';
      }

        addStationMarkers(stations);
    });

    // Add error handling for HTMX requests
    document.body.addEventListener('htmx:responseError', function(event) {
        if (event.detail.target.id === 'station-data') {
            const statusDiv = document.getElementById('prediction-status');
            if (statusDiv) {
                statusDiv.textContent = '❌ Failed to load data - showing current view';
            }
            // Don't clear existing markers on error
        }
    });
}
  
  document.addEventListener('DOMContentLoaded', () => {
    initMap();
    setupHTMX();
    updateAutoRefreshMode(); // Set correct mode for auto-refresh on page load
  });