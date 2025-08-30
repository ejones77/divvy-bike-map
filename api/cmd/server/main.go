package main

import (
	"context"
	"flag"
	"log"
	"os"
	"path/filepath"
	"sort"

	"api/internal"

	"github.com/joho/godotenv"
)

func runMigrations(db *internal.Database) error {
	migrationsDir := "./migrations"

	if _, err := os.Stat(migrationsDir); os.IsNotExist(err) {
		log.Println("No migrations directory found, skipping migrations")
		return nil
	}

	files, err := filepath.Glob(filepath.Join(migrationsDir, "*.sql"))
	if err != nil {
		return err
	}

	if len(files) == 0 {
		log.Println("No migration files found")
		return nil
	}

	sort.Strings(files)

	log.Printf("Running %d migration files...", len(files))
	for _, file := range files {
		log.Printf("Executing migration: %s", filepath.Base(file))

		content, err := os.ReadFile(file)
		if err != nil {
			return err
		}

		if err := db.ExecMigration(context.Background(), string(content)); err != nil {
			return err
		}
	}

	log.Println("All migrations completed successfully")
	return nil
}

func main() {
	migrateOnly := flag.Bool("migrate", false, "Run migrations only and exit")
	flag.Parse()

	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, using environment variables")
	}

	config := internal.LoadConfig()

	if err := config.Validate(); err != nil {
		log.Fatal("Configuration validation failed:", err)
	}

	database, err := internal.NewDatabase(config)
	if err != nil {
		log.Fatal("Failed to initialize database:", err)
	}
	defer database.Close()

	if err := runMigrations(database); err != nil {
		log.Fatal("Failed to run migrations:", err)
	}

	if *migrateOnly {
		log.Println("Migrations completed, exiting")
		return
	}

	divvyClient := internal.NewDivvyClient(config)

	handlers := internal.NewHTTPHandlers(database, divvyClient, config)

	// AUTO-REFRESH DATA ON STARTUP
	log.Println("Refreshing station data on startup in background...")
	go func() {
		if err := handlers.RefreshStationDataInternal(context.Background()); err != nil {
			log.Printf("Failed to refresh station data: %v", err)
			return
		}
		log.Println("Station data refresh completed")
	}()

	server, err := internal.NewServer(config, handlers)
	if err != nil {
		log.Fatal("Failed to create server:", err)
	}

	if err := server.Start(); err != nil {
		log.Fatal("Server failed:", err)
	}
}
