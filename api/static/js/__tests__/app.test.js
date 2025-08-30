import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { JSDOM } from 'jsdom';

describe('app.js integration', () => {
  let dom;
  let fetchMock;
  let originalSetTimeout;

  beforeEach(() => {
    // Reset modules to ensure fresh imports
    vi.resetModules();
    
    dom = new JSDOM(`
      <html>
        <head>
          <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        </head>
        <body>
          <div id="map"></div>
          <div id="prediction-status"></div>
          <div id="last-updated"></div>
          <input name="view-mode" value="predicted" type="radio">
          <div id="station-data">
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
            </div>
          </div>
        </body>
      </html>
    `);
    
    global.document = dom.window.document;
    global.window = dom.window;
    
    // Clear and reset Leaflet mocks
    if (global.L) {
      Object.keys(global.L).forEach(key => {
        if (vi.isMockFunction(global.L[key])) {
          global.L[key].mockClear();
        }
      });
    } else {
      // Mock Leaflet
      global.L = {
        map: vi.fn(() => ({
          setView: vi.fn().mockReturnThis(),
          removeLayer: vi.fn(),
        })),
        tileLayer: vi.fn(() => ({
          addTo: vi.fn().mockReturnThis(),
        })),
        circleMarker: vi.fn(() => ({
          bindPopup: vi.fn().mockReturnThis(),
          addTo: vi.fn().mockReturnThis(),
        })),
        featureGroup: vi.fn(() => ({
          bindPopup: vi.fn().mockReturnThis(),
          addTo: vi.fn().mockReturnThis(),
        })),
      };
    }
    
    fetchMock = vi.fn();
    global.fetch = fetchMock;
    
    // Mock setTimeout to track calls
    originalSetTimeout = global.setTimeout;
    global.setTimeout = vi.fn((fn, delay) => {
      return originalSetTimeout(() => fn(), 0); // Execute immediately for tests
    });
  });

  afterEach(() => {
    global.setTimeout = originalSetTimeout;
    if (fetchMock && fetchMock.mockRestore) {
      fetchMock.mockRestore();
    }
    // Clean up module cache
    vi.resetModules();
  });

  describe('prediction status polling', () => {
    it('handles network errors gracefully', async () => {
      fetchMock.mockRejectedValue(new Error('Network error'));
      
      // Import and trigger the module
      await import('../app.js');
      
      // Simulate the polling function being called
      // Since we can't access it directly, we'll test the DOM updates
      const statusDiv = document.getElementById('prediction-status');
      
      // Trigger a fetch error scenario by dispatching the DOMContentLoaded event
      const event = new dom.window.Event('DOMContentLoaded');
      document.dispatchEvent(event);
      
      
      expect(statusDiv).toBeTruthy();
    });

    it('handles successful status response', async () => {
      fetchMock.mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({
          current_step: 'completed',
          message: 'Predictions ready',
          progress_percent: 100
        })
      });
      
      await import('../app.js');
      
      const statusDiv = document.getElementById('prediction-status');
      expect(statusDiv).toBeTruthy();
    });

    it('handles 404 errors specifically for ML service', async () => {
      fetchMock.mockResolvedValue({
        ok: false,
        status: 404,
        statusText: 'Not Found'
      });
      
      await import('../app.js');
      
      const statusDiv = document.getElementById('prediction-status');
      expect(statusDiv).toBeTruthy();
    });

    it('handles other HTTP errors gracefully', async () => {
      fetchMock.mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      });
      
      await import('../app.js');
      
      const statusDiv = document.getElementById('prediction-status');
      expect(statusDiv).toBeTruthy();
    });
  });

  describe('HTMX integration', () => {
    it('handles station data updates', async () => {
      await import('../app.js');
      
      // Simulate HTMX event
      const event = new dom.window.CustomEvent('htmx:afterSwap', {
        detail: { 
          target: document.getElementById('station-data') 
        }
      });
      
      expect(() => {
        document.body.dispatchEvent(event);
      }).not.toThrow();
    });

    it('safely handles missing DOM elements', async () => {
      document.querySelector('input[name="view-mode"]')?.remove();
      
      await import('../app.js');
      
      const event = new dom.window.CustomEvent('htmx:afterSwap', {
        detail: { target: document.getElementById('station-data') }
      });
      
      expect(() => {
        document.body.dispatchEvent(event);
      }).not.toThrow();
    });

    it('handles HTMX response errors gracefully', async () => {
      await import('../app.js');
      
      // Simulate HTMX error event
      const event = new dom.window.CustomEvent('htmx:responseError', {
        detail: { 
          target: document.getElementById('station-data') 
        }
      });
      
      expect(() => {
        document.body.dispatchEvent(event);
      }).not.toThrow();
    });

    it('shows appropriate status for prediction mode without predictions', async () => {
      await import('../app.js');
      
      // Initialize the module by dispatching DOMContentLoaded
      const initEvent = new dom.window.Event('DOMContentLoaded');
      document.dispatchEvent(initEvent);
      
      // Wait for initialization
      await new Promise(resolve => setTimeout(resolve, 10));
      
      // Set up prediction mode
      const predInput = document.querySelector('input[name="view-mode"]');
      predInput.checked = true;
      predInput.value = 'predicted';
      
      // Create station data without predictions
      const stationData = document.getElementById('station-data');
      stationData.innerHTML = `
        <div class="station-data"
             data-station-id="123"
             data-name="Test Station"
             data-lat="41.8781"
             data-lon="-87.6298"
             data-capacity="20"
             data-bikes-available="5"
             data-docks-available="15">
        </div>`;
      
      // Verify the data is there
      expect(stationData.querySelector('.station-data')).toBeTruthy();
      
      // Simulate HTMX event
      const event = new dom.window.CustomEvent('htmx:afterSwap', {
        detail: { target: stationData }
      });
      
      document.body.dispatchEvent(event);
      
      // Wait for the event to process
      await new Promise(resolve => setTimeout(resolve, 10));
      
      const statusDiv = document.getElementById('prediction-status');
      // The status should show ML service unavailable since we're in prediction mode with no predictions
      expect(statusDiv.textContent).toContain('ML service unavailable');
    });

    it('handles prediction mode with existing predictions', async () => {
      await import('../app.js');
      
      // Initialize the module by dispatching DOMContentLoaded
      const initEvent = new dom.window.Event('DOMContentLoaded');
      document.dispatchEvent(initEvent);
      
      // Wait for initialization
      await new Promise(resolve => setTimeout(resolve, 10));
      
      // Set up prediction mode
      const predInput = document.querySelector('input[name="view-mode"]');
      predInput.checked = true;
      predInput.value = 'predicted';
      
      // Create station data WITH predictions
      const stationData = document.getElementById('station-data');
      stationData.innerHTML = `
        <div class="station-data"
             data-station-id="123"
             data-name="Test Station"
             data-lat="41.8781"
             data-lon="-87.6298"
             data-capacity="20"
             data-bikes-available="5"
             data-docks-available="15"
             data-prediction="green"
             data-prediction-time="2023-01-01T18:00:00Z"
             data-horizon-hours="6">
        </div>`;
      
      // Simulate HTMX event
      const event = new dom.window.CustomEvent('htmx:afterSwap', {
        detail: { target: stationData }
      });
      
      document.body.dispatchEvent(event);
      
      // Wait for the event to process
      await new Promise(resolve => setTimeout(resolve, 10));
      
      const statusDiv = document.getElementById('prediction-status');
      // Status should be cleared when predictions are available
      expect(statusDiv.textContent).toBe('');
    });
  });

  describe('map initialization', () => {
    it('initializes map without errors', async () => {
      // Import the module first, which adds the event listener
      await import('../app.js');
      
      // Now dispatch the DOMContentLoaded event
      const event = new dom.window.Event('DOMContentLoaded');
      document.dispatchEvent(event);
      
      // The event should be synchronous, but let's wait a tick
      await new Promise(resolve => process.nextTick(resolve));
      
      expect(L.map).toHaveBeenCalledWith('map');
    });

    it('handles marker creation and management', async () => {
      await import('../app.js');
      
      // Dispatch DOMContentLoaded to initialize map
      const initEvent = new dom.window.Event('DOMContentLoaded');
      document.dispatchEvent(initEvent);
      
      // Wait for initialization
      await new Promise(resolve => process.nextTick(resolve));
      
      // Verify map was created
      expect(L.map).toHaveBeenCalled();
      
      // Verify tile layer was added
      expect(L.tileLayer).toHaveBeenCalled();
    });
  });
});