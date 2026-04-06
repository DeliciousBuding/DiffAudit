package api

import (
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"sort"
)

type Config struct {
	ExperimentsRoot string
	JobsRoot        string
}

type Server struct {
	config Config
	mux    *http.ServeMux
}

type modelOption struct {
	Key         string  `json:"key"`
	Label       string  `json:"label"`
	AccessLevel string  `json:"access_level"`
	Availability string `json:"availability"`
	Paper       string  `json:"paper"`
	Method      string  `json:"method"`
	Backend     string  `json:"backend"`
	Scheduler   *string `json:"scheduler"`
}

var models = []modelOption{
	{
		Key:          "sd15-ddim",
		Label:        "Stable Diffusion 1.5 + DDIM",
		AccessLevel:  "black-box",
		Availability: "ready",
		Paper:        "BlackBox_Reconstruction_ArXiv2023",
		Method:       "recon",
		Backend:      "stable_diffusion",
		Scheduler:    stringPtr("ddim"),
	},
	{
		Key:          "kandinsky-v22",
		Label:        "Kandinsky v2.2",
		AccessLevel:  "black-box",
		Availability: "partial",
		Paper:        "BlackBox_Reconstruction_ArXiv2023",
		Method:       "recon",
		Backend:      "kandinsky_v22",
	},
	{
		Key:          "dit-xl2-256",
		Label:        "DiT-XL/2 256",
		AccessLevel:  "black-box",
		Availability: "partial",
		Paper:        "Scalable_Diffusion_Transformers_2022",
		Method:       "sample",
		Backend:      "dit",
	},
}

func stringPtr(value string) *string {
	return &value
}

func NewServer(config Config) *Server {
	mux := http.NewServeMux()
	server := &Server{
		config: config,
		mux:    mux,
	}
	mux.HandleFunc("GET /health", server.handleHealth)
	mux.HandleFunc("GET /api/v1/models", server.handleModels)
	mux.HandleFunc("GET /api/v1/experiments/recon/best", server.handleBestRecon)
	mux.HandleFunc("GET /api/v1/experiments/{workspace}/summary", server.handleWorkspaceSummary)
	return server
}

func (s *Server) Handler() http.Handler {
	return s.mux
}

func (s *Server) handleHealth(writer http.ResponseWriter, _ *http.Request) {
	writeJSON(writer, http.StatusOK, map[string]any{
		"status":           "ok",
		"experiments_root": s.config.ExperimentsRoot,
		"jobs_root":        s.config.JobsRoot,
	})
}

func (s *Server) handleModels(writer http.ResponseWriter, _ *http.Request) {
	writeJSON(writer, http.StatusOK, models)
}

func (s *Server) handleBestRecon(writer http.ResponseWriter, _ *http.Request) {
	summaryPath, err := s.bestReconSummaryPath()
	if err != nil {
		writeError(writer, http.StatusNotFound, err.Error())
		return
	}
	s.handleSummaryPath(writer, summaryPath)
}

func (s *Server) handleWorkspaceSummary(writer http.ResponseWriter, request *http.Request) {
	workspace := request.PathValue("workspace")
	if workspace == "" {
		writeError(writer, http.StatusBadRequest, "workspace is required")
		return
	}
	summaryPath := filepath.Join(s.config.ExperimentsRoot, workspace, "summary.json")
	s.handleSummaryPath(writer, summaryPath)
}

func (s *Server) handleSummaryPath(writer http.ResponseWriter, summaryPath string) {
	payload, err := readJSONFile(summaryPath)
	if err != nil {
		writeError(writer, http.StatusNotFound, err.Error())
		return
	}
	writeJSON(writer, http.StatusOK, summaryEnvelope(summaryPath, payload))
}

func (s *Server) bestReconSummaryPath() (string, error) {
	statusPath := filepath.Join(s.config.ExperimentsRoot, "blackbox-status", "summary.json")
	if payload, err := readJSONFile(statusPath); err == nil {
		methods, ok := payload["methods"].(map[string]any)
		if ok {
			recon, ok := methods["recon"].(map[string]any)
			if ok {
				bestPath, ok := recon["best_evidence_path"].(string)
				if ok && bestPath != "" {
					if _, err := os.Stat(bestPath); err == nil {
						return bestPath, nil
					}
				}
			}
		}
	}

	pattern := filepath.Join(s.config.ExperimentsRoot, "*", "summary.json")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return "", err
	}

	type candidate struct {
		path       string
		sampleCount float64
	}

	candidates := make([]candidate, 0)
	for _, path := range matches {
		payload, err := readJSONFile(path)
		if err != nil {
			continue
		}
		method, _ := payload["method"].(string)
		if method != "recon" {
			continue
		}
		sampleCount := extractSampleCount(payload)
		candidates = append(candidates, candidate{path: path, sampleCount: sampleCount})
	}
	if len(candidates) == 0 {
		return "", errors.New("no recon experiment summaries found")
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].sampleCount == candidates[j].sampleCount {
			return candidates[i].path < candidates[j].path
		}
		return candidates[i].sampleCount < candidates[j].sampleCount
	})
	return candidates[len(candidates)-1].path, nil
}

func extractSampleCount(payload map[string]any) float64 {
	artifacts, ok := payload["artifacts"].(map[string]any)
	if !ok {
		return 0
	}
	total := 0.0
	for _, value := range artifacts {
		artifact, ok := value.(map[string]any)
		if !ok {
			continue
		}
		sampleCount, ok := artifact["sample_count"].(float64)
		if ok {
			total += sampleCount
		}
	}
	return total
}

func summaryEnvelope(summaryPath string, payload map[string]any) map[string]any {
	backend := any(nil)
	scheduler := any(nil)
	if runtime, ok := payload["runtime"].(map[string]any); ok {
		backend = runtime["backend"]
		scheduler = runtime["scheduler"]
	}
	return map[string]any{
		"status":         payload["status"],
		"paper":          payload["paper"],
		"method":         payload["method"],
		"mode":           payload["mode"],
		"backend":        backend,
		"scheduler":      scheduler,
		"workspace":      payload["workspace"],
		"metrics":        payload["metrics"],
		"artifact_paths": payload["artifact_paths"],
		"summary_path":   summaryPath,
		"raw":            payload,
	}
}

func readJSONFile(path string) (map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, err
	}
	return payload, nil
}

func writeJSON(writer http.ResponseWriter, statusCode int, payload any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(statusCode)
	_ = json.NewEncoder(writer).Encode(payload)
}

func writeError(writer http.ResponseWriter, statusCode int, detail string) {
	writeJSON(writer, statusCode, map[string]any{"detail": detail})
}
