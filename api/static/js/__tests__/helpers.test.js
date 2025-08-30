import { describe, it, expect, vi, beforeEach } from 'vitest';
import { JSDOM } from 'jsdom';
import { 
  getAvailabilityColor, 
  getPredictionColor, 
  parseStations,
  createSingleMarker,
  createDualMarker,
  formatPredictionTime
} from '../helpers.js';

// Mock Leaflet globally
global.L = {
  circleMarker: vi.fn(() => ({ mockMarker: true })),
  featureGroup: vi.fn((markers) => ({ 
    mockFeatureGroup: true, 
    markers 
  }))
};

describe('helpers', () => {
  let dom;
  
  beforeEach(() => {
    vi.clearAllMocks();
    
    dom = new JSDOM('<!DOCTYPE html><html><body></body></html>');
    global.document = dom.window.document;
    global.window = dom.window;
  });

  describe('getAvailabilityColor', () => {
    it('returns red for 0 bikes', () => {
      expect(getAvailabilityColor(0)).toBe('#ef4444');
    });
    
    it('returns yellow for low availability', () => {
      expect(getAvailabilityColor(1)).toBe('#f59e0b');
      expect(getAvailabilityColor(3)).toBe('#f59e0b');
    });
    
    it('returns green for good availability', () => {
      expect(getAvailabilityColor(4)).toBe('#10b981');
      expect(getAvailabilityColor(10)).toBe('#10b981');
    });
  });

  describe('getPredictionColor', () => {
    it('maps prediction strings to colors', () => {
      expect(getPredictionColor('red')).toBe('#ef4444');
      expect(getPredictionColor('yellow')).toBe('#f59e0b');
      expect(getPredictionColor('green')).toBe('#10b981');
    });
    
    it('returns gray for unknown predictions', () => {
      expect(getPredictionColor('unknown')).toBe('#6b7280');
      expect(getPredictionColor('')).toBe('#6b7280');
      expect(getPredictionColor(null)).toBe('#6b7280');
    });
  });

  describe('createSingleMarker', () => {
    it('creates marker with correct properties', () => {
      const station = { lat: 41.8781, lon: -87.6298 };
      const color = '#10b981';
      
      createSingleMarker(station, color);
      
      expect(L.circleMarker).toHaveBeenCalledWith(
        [41.8781, -87.6298],
        expect.objectContaining({
          radius: 8,
          fillColor: '#10b981',
          fillOpacity: 0.8
        })
      );
    });
  });

  describe('createDualMarker', () => {
    it('creates feature group with two markers', () => {
      const station = { lat: 41.8781, lon: -87.6298 };
      
      createDualMarker(station, '#10b981', '#ef4444');
      
      expect(L.circleMarker).toHaveBeenCalledTimes(2);
      expect(L.featureGroup).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({ mockMarker: true }),
          expect.objectContaining({ mockMarker: true })
        ])
      );
    });
  });

  describe('parseStations', () => {
    it('parses complete station data', () => {
      const div = document.createElement('div');
      div.innerHTML = `
        <div class="station-data"
             data-station-id="123"
             data-name="Test Station"
             data-lat="41.8781"
             data-lon="-87.6298"
             data-capacity="20"
             data-bikes-available="5"
             data-docks-available="15"
             data-prediction="green"
             data-prediction-time="2023-01-01T12:00:00Z"
             data-horizon-hours="6">
        </div>`;
      
      const result = parseStations(div);
      
      expect(result[0]).toEqual({
        station_id: '123',
        name: 'Test Station',
        lat: 41.8781,
        lon: -87.6298,
        capacity: 20,
        num_bikes_available: 5,
        num_docks_available: 15,
        predicted_availability: 'green',
        prediction_time: '2023-01-01T12:00:00Z',
        horizon_hours: 6
      });
    });

    it('handles missing prediction data', () => {
      const div = document.createElement('div');
      div.innerHTML = `
        <div class="station-data"
             data-station-id="123"
             data-name="Test"
             data-lat="1.1"
             data-lon="2.2"
             data-capacity="10"
             data-bikes-available="3"
             data-docks-available="7">
        </div>`;
      
      const result = parseStations(div);
      
      expect(result[0]).toMatchObject({
        predicted_availability: null,
        prediction_time: null,
        horizon_hours: null
      });
    });

    it('handles invalid numeric data', () => {
      const div = document.createElement('div');
      div.innerHTML = `
        <div class="station-data"
             data-lat="invalid"
             data-bikes-available="not-a-number"
             data-horizon-hours="bad">
        </div>`;
      
      const result = parseStations(div);
      
      expect(result[0]).toMatchObject({
        lat: NaN, // parseFloat('invalid') = NaN
        num_bikes_available: NaN, // parseInt('not-a-number') = NaN
        horizon_hours: null // parseInt('bad') is falsy, so null
      });
    });
  });

  describe('formatPredictionTime', () => {
    it('formats valid prediction time (already future time)', () => {
      const result = formatPredictionTime('2023-01-01T18:00:00Z', 6);
      
      // Should format to a valid time (local time conversion may vary)
      expect(result).toMatch(/^\d{1,2}:\d{2}.*[AP]M$/);
    });

    it('handles invalid prediction time', () => {
      const result = formatPredictionTime('invalid-date', 2);
      
      // Should still return a formatted time (using current time, no offset)
      expect(result).toMatch(/^\d{1,2}:\d{2}.*[AP]M$/);
    });

    it('handles missing horizon hours', () => {
      const result = formatPredictionTime('2023-01-01T12:00:00Z', null);
      
      // Should use prediction time directly (local time conversion)
      expect(result).toMatch(/^\d{1,2}:\d{2}.*[AP]M$/);
    });

    it('formats time correctly for different hours', () => {
      const result = formatPredictionTime('2023-01-01T00:00:00Z', 6);
      
      // Should format to a valid time
      expect(result).toMatch(/^\d{1,2}:\d{2}.*[AP]M$/);
    });

    it('handles edge case of 12 PM', () => {
      const result = formatPredictionTime('2023-01-01T12:00:00Z', 6);
      
      // Should format to a valid time
      expect(result).toMatch(/^\d{1,2}:\d{2}.*[AP]M$/);
    });
  });
});
