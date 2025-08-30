package internal

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"golang.org/x/sync/errgroup"
)

type DivvyClient struct {
	stationInfoURL   string
	stationStatusURL string
	httpClient       *http.Client
}

func NewDivvyClient(cfg *Config) *DivvyClient {
	return &DivvyClient{
		stationInfoURL:   cfg.Divvy.StationInfoURL,
		stationStatusURL: cfg.Divvy.StationStatusURL,
		httpClient:       &http.Client{Timeout: 30 * time.Second},
	}
}

func (c *DivvyClient) fetchJSON(ctx context.Context, url string, target interface{}) error {
    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return fmt.Errorf("create request: %w", err)
    }

    resp, err := c.httpClient.Do(req)
    if err != nil {
        return fmt.Errorf("http request: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
    }

    if err := json.NewDecoder(resp.Body).Decode(target); err != nil {
        return fmt.Errorf("decode JSON: %w", err)
    }

    return nil
}

func (c *DivvyClient) FetchStationData(ctx context.Context) ([]DivvyStation, []DivvyStationStatus, error) {
    var stationInfo DivvyStationInfoResponse
    var stationStatus DivvyStationStatusResponse

    g, ctx := errgroup.WithContext(ctx)

    g.Go(func() error {
        return c.fetchJSON(ctx, c.stationInfoURL, &stationInfo)
    })

    g.Go(func() error {
        return c.fetchJSON(ctx, c.stationStatusURL, &stationStatus)
    })

    if err := g.Wait(); err != nil {
        return nil, nil, fmt.Errorf("failed to fetch station data: %w", err)
    }

    log.Printf("Fetched data for %d stations", len(stationInfo.Data.Stations))
    return stationInfo.Data.Stations, stationStatus.Data.Stations, nil
}
