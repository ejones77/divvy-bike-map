package internal

import (
	"context"
	"fmt"
	"log"
)

type StationService struct {
	database    DatabaseInterface
	divvyClient DivvyClientInterface
}

func NewStationService(database DatabaseInterface, divvyClient DivvyClientInterface) *StationService {
	return &StationService{
		database:    database,
		divvyClient: divvyClient,
	}
}

func (s *StationService) RefreshStationData(ctx context.Context) error {
	stations, statuses, err := s.divvyClient.FetchStationData(ctx)
	if err != nil {
		return err
	}

	dbStations := make([]Station, len(stations))
	for i, divvyStation := range stations {
		dbStations[i] = s.convertToStation(divvyStation)
	}

	availabilities := make([]StationAvailability, len(statuses))
	for i, divvyStatus := range statuses {
		availabilities[i] = s.convertToAvailability(divvyStatus)
	}

	if err := s.database.UpsertStations(ctx, dbStations); err != nil {
		return fmt.Errorf("failed to store stations: %w", err)
	}

	if err := s.database.InsertAvailabilities(ctx, availabilities); err != nil {
		return fmt.Errorf("failed to store availabilities: %w", err)
	}

	log.Printf("Stored data for %d stations", len(stations))
	return nil
}

func (s *StationService) convertToStation(divvyStation DivvyStation) Station {
	return Station{
		StationID: divvyStation.StationID,
		Name:      divvyStation.Name,
		Lat:       divvyStation.Lat,
		Lon:       divvyStation.Lon,
		Capacity:  divvyStation.Capacity,
	}
}

func (s *StationService) convertToAvailability(divvyStatus DivvyStationStatus) StationAvailability {
	return StationAvailability{
		StationID:         divvyStatus.StationID,
		NumBikesAvailable: divvyStatus.NumBikesAvailable,
		NumDocksAvailable: divvyStatus.NumDocksAvailable,
		IsInstalled:       divvyStatus.IsInstalled,
		IsRenting:         divvyStatus.IsRenting,
		IsReturning:       divvyStatus.IsReturning,
		LastReported:      divvyStatus.LastReported,
	}
}
