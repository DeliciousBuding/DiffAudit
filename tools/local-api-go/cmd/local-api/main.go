package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"diffaudit/local-api-go/internal/api"
)

type runtimeConfig struct {
	Host            string
	Port            string
	ExperimentsRoot string
	JobsRoot        string
	ProjectRoot     string
}

func parseConfig(args []string) (runtimeConfig, error) {
	flagSet := flag.NewFlagSet("local-api", flag.ContinueOnError)
	flagSet.SetOutput(os.Stdout)

	projectRootDefault := detectProjectRoot(defaultPath("."))
	host := flagSet.String("host", "127.0.0.1", "listen host")
	port := flagSet.String("port", "8765", "listen port")
	experimentsRoot := flagSet.String("experiments-root", filepath.Join(projectRootDefault, "experiments"), "experiments root")
	jobsRoot := flagSet.String("jobs-root", filepath.Join(projectRootDefault, "workspaces", "local-api", "jobs"), "jobs root")
	projectRoot := flagSet.String("project-root", projectRootDefault, "project root")

	if err := flagSet.Parse(args); err != nil {
		return runtimeConfig{}, err
	}

	return runtimeConfig{
		Host:            *host,
		Port:            *port,
		ExperimentsRoot: *experimentsRoot,
		JobsRoot:        *jobsRoot,
		ProjectRoot:     *projectRoot,
	}, nil
}

func defaultPath(relative string) string {
	current, err := os.Getwd()
	if err != nil {
		return relative
	}
	if relative == "." {
		return current
	}
	return filepath.Clean(filepath.Join(current, relative))
}

func detectProjectRoot(current string) string {
	cleaned := filepath.Clean(current)
	toolPath := filepath.Join("tools", "local-api-go")
	if strings.HasSuffix(cleaned, toolPath) {
		return filepath.Clean(filepath.Join(cleaned, "..", ".."))
	}
	return cleaned
}

func main() {
	config, err := parseConfig(os.Args[1:])
	if err != nil {
		os.Exit(2)
	}

	server := api.NewServer(api.Config{
		ExperimentsRoot: config.ExperimentsRoot,
		JobsRoot:        config.JobsRoot,
		ProjectRoot:     config.ProjectRoot,
		AutoStartJobs:   true,
	})

	address := fmt.Sprintf("%s:%s", config.Host, config.Port)
	if err := http.ListenAndServe(address, server.Handler()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
