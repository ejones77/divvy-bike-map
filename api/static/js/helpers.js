// ES module: pure helpers
export function getAvailabilityColor(numBikes) {
    if (numBikes === 0) return '#ef4444';
    if (numBikes <= 3) return '#f59e0b';
    return '#10b981';
  }
  
  export function getPredictionColor(prediction) {
    switch (prediction) {
      case 'red': return '#ef4444';
      case 'yellow': return '#f59e0b';
      case 'green': return '#10b981';
      default: return '#6b7280';
    }
  }
  
  export function createSingleMarker(station, color) {
    return L.circleMarker([station.lat, station.lon], {
      radius: 8,
      fillColor: color,
      color: '#ffffff',
      weight: 2,
      opacity: 1,
      fillOpacity: 0.8
    });
  }
  
  export function createDualMarker(station, currentColor, predictionColor) {
    const outerMarker = L.circleMarker([station.lat, station.lon], {
      radius: 12,
      fillColor: predictionColor,
      color: '#ffffff',
      weight: 2,
      opacity: 1,
      fillOpacity: 0.5
    });
    const innerMarker = L.circleMarker([station.lat, station.lon], {
      radius: 6,
      fillColor: currentColor,
      color: '#ffffff',
      weight: 1,
      opacity: 1,
      fillOpacity: 0.9
    });
    // Use featureGroup so the group can handle events (e.g., click) and propagate popups.
    return L.featureGroup([outerMarker, innerMarker]);
  }
  
  export function parseStations(containerEl) {
    const stationElements = containerEl.querySelectorAll('.station-data');
    return Array.from(stationElements).map(el => ({
      station_id: el.dataset.stationId,
      name: el.dataset.name,
      lat: parseFloat(el.dataset.lat),
      lon: parseFloat(el.dataset.lon),
      capacity: parseInt(el.dataset.capacity),
      num_bikes_available: parseInt(el.dataset.bikesAvailable),
      num_docks_available: parseInt(el.dataset.docksAvailable),
      predicted_availability: el.dataset.prediction || null,
      prediction_time: el.dataset.predictionTime || null,
      horizon_hours: (() => {
        const val = parseInt(el.dataset.horizonHours);
        return Number.isFinite(val) ? val : null;
      })()
    }));
  }
  
  export function formatPredictionTime(predictionTime, horizonHours) {
    let predDate;
    
    if (predictionTime) {
      predDate = new Date(predictionTime);
      // If the date is invalid, fall back to current time
      if (isNaN(predDate.getTime())) {
        predDate = new Date();
      }
    } else {
      predDate = new Date();
    }
    
    // The prediction time from the ML pipeline should already be the future time
    // So we don't need to add horizon hours again
    const targetTime = predDate;
    
    // Use local time methods to display in user's timezone
    const localHours = targetTime.getHours();
    const localMinutes = targetTime.getMinutes();
    
    // Convert to 12-hour format
    const displayHours = localHours === 0 ? 12 : localHours > 12 ? localHours - 12 : localHours;
    const ampm = localHours >= 12 ? 'PM' : 'AM';
    const minutes = localMinutes.toString().padStart(2, '0');
    
    return `${displayHours}:${minutes} ${ampm}`;
  }