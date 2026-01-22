package config

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
)

type envVar struct {
	Name  string
	Value string
}

type integrationDef struct {
	Name             string
	DisplayName      string
	Command          string
	EnvVars          func(model string) []envVar
	Args             func(model string) []string
	Setup            func(models []string) error
	CheckInstall     func() error
	ConfigPaths      func() []string // paths that will be modified
	ConfiguredModels func() []string // models already configured
}

// configPaths returns the config file paths for this integration, or nil if none.
func (d *integrationDef) configPaths() []string {
	if d.ConfigPaths == nil {
		return nil
	}
	return d.ConfigPaths()
}

// configuredModels returns the models already configured for this integration, or nil if none.
func (d *integrationDef) configuredModels() []string {
	if d.ConfiguredModels == nil {
		return nil
	}
	return d.ConfiguredModels()
}

// checkCommand returns an error if the command is not installed
func checkCommand(cmd, installInstructions string) func() error {
	return func() error {
		if _, err := exec.LookPath(cmd); err != nil {
			return fmt.Errorf("%s is not installed, %s", cmd, installInstructions)
		}
		return nil
	}
}

var integrationRegistry = map[string]*integrationDef{
	"claude":   claudeIntegration,
	"codex":    codexIntegration,
	"droid":    droidIntegration,
	"opencode": openCodeIntegration,
}

func integration(name string) (*integrationDef, bool) {
	i, ok := integrationRegistry[strings.ToLower(name)]
	return i, ok
}

func integrationConfiguredModels(integ *integrationDef, integrationName string) []string {
	// Get models that exist in the integration's config
	var integrationModels []string
	if integ != nil {
		integrationModels = integ.configuredModels()
	}

	// Get our saved integration config for the correct order (default first)
	savedConfig, err := loadIntegration(integrationName)
	if err != nil || len(savedConfig.Models) == 0 {
		return integrationModels
	}
	if len(integrationModels) == 0 {
		return savedConfig.Models
	}

	// Merge: saved order first (filtered to still-existing models), then any new models
	integrationModelSet := make(map[string]bool, len(integrationModels))
	for _, m := range integrationModels {
		integrationModelSet[m] = true
	}

	seen := make(map[string]bool, len(savedConfig.Models))
	var result []string
	for _, m := range savedConfig.Models {
		if integrationModelSet[m] {
			result = append(result, m)
			seen[m] = true
		}
	}
	for _, m := range integrationModels {
		if !seen[m] {
			result = append(result, m)
		}
	}

	return result
}

func sortedIntegrationNames() []string {
	names := slices.Collect(maps.Keys(integrationRegistry))
	slices.Sort(names)
	return names
}

func selectIntegration() (string, error) {
	if len(integrationRegistry) == 0 {
		return "", fmt.Errorf("no integrations available")
	}

	var items []selectItem
	for _, name := range sortedIntegrationNames() {
		integration := integrationRegistry[name]
		description := integration.DisplayName
		if conn, err := loadIntegration(name); err == nil && conn.defaultModel() != "" {
			description = fmt.Sprintf("%s (%s)", integration.DisplayName, conn.defaultModel())
		}
		items = append(items, selectItem{Name: integration.Name, Description: description})
	}

	return selectPrompt("Select integration:", items)
}

func selectModels(ctx context.Context, integrationName, currentModel string) ([]string, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}

	models, err := client.List(ctx)
	if err != nil {
		return nil, err
	}

	if len(models.Models) == 0 {
		return nil, fmt.Errorf("no models available, run 'ollama pull <model>' first")
	}

	var items []selectItem
	cloudModels := make(map[string]bool)
	for _, m := range models.Models {
		if m.RemoteModel != "" {
			cloudModels[m.Name] = true
		}
		items = append(items, selectItem{Name: m.Name})
	}

	if len(items) == 0 {
		return nil, fmt.Errorf("no local models available, run 'ollama pull <model>' first")
	}

	integ, _ := integration(integrationName)
	preChecked := integrationConfiguredModels(integ, integrationName)
	preCheckedSet := make(map[string]bool)
	for _, name := range preChecked {
		preCheckedSet[name] = true
	}

	// Resolve currentModel to full name from list (e.g., "llama3.2" -> "llama3.2:latest")
	if currentModel != "" {
		for _, item := range items {
			if item.Name == currentModel || strings.HasPrefix(item.Name, currentModel+":") {
				currentModel = item.Name
				break
			}
		}
	}

	// If currentModel is already configured, move it to front of preChecked
	if currentModel != "" && preCheckedSet[currentModel] {
		newPreChecked := []string{currentModel}
		for _, m := range preChecked {
			if m != currentModel {
				newPreChecked = append(newPreChecked, m)
			}
		}
		preChecked = newPreChecked
	}

	slices.SortFunc(items, func(a, b selectItem) int {
		aName, bName := a.Name, b.Name
		aChecked, bChecked := preCheckedSet[aName], preCheckedSet[bName]
		aCurrent, bCurrent := aName == currentModel, bName == currentModel

		// Current model comes first
		if aCurrent != bCurrent {
			if aChecked || bChecked {
				if aCurrent && aChecked {
					return -1
				}
				return 1
			}
			if aCurrent {
				return -1
			}
			return 1
		}
		// Pre-checked models come before unchecked
		if aChecked != bChecked {
			if aChecked {
				return -1
			}
			return 1
		}
		// Alphabetical within groups
		return strings.Compare(strings.ToLower(aName), strings.ToLower(bName))
	})

	supportsMultiModel := integ != nil && integ.Setup != nil

	var selected []string
	if supportsMultiModel {
		selected, err = multiSelectPrompt(fmt.Sprintf("Select models for %s:", integ.DisplayName), items, preChecked)
		if err != nil {
			return nil, err
		}
	} else {
		model, err := selectPrompt(fmt.Sprintf("Select model for %s:", integ.DisplayName), items)
		if err != nil {
			return nil, err
		}
		selected = []string{model}
	}

	for _, model := range selected {
		if cloudModels[model] {
			if err := ensureSignedIn(ctx, client); err != nil {
				return nil, err
			}
			break
		}
	}

	return selected, nil
}

func ensureSignedIn(ctx context.Context, client *api.Client) error {
	user, err := client.Whoami(ctx)
	if err == nil && user != nil && user.Name != "" {
		return nil
	}

	var aErr api.AuthorizationError
	if !errors.As(err, &aErr) || aErr.SigninURL == "" {
		return err
	}

	yes, err := confirmPrompt("Sign in to ollama.com?")
	if err != nil || !yes {
		return fmt.Errorf("sign in required for cloud models")
	}

	fmt.Fprintf(os.Stderr, "\nTo sign in, navigate to:\n    %s\n\n", aErr.SigninURL)
	fmt.Fprintf(os.Stderr, "\033[90mwaiting for sign in to complete...\033[0m")

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			fmt.Fprintf(os.Stderr, "\n")
			return ctx.Err()
		case <-ticker.C:
			user, err := client.Whoami(ctx)
			if err == nil && user != nil && user.Name != "" {
				fmt.Fprintf(os.Stderr, "\r\033[K\033[A\r\033[K\033[1msigned in:\033[0m %s\n", user.Name)
				return nil
			}
			fmt.Fprintf(os.Stderr, ".")
		}
	}
}

func hasLocalModel(models []string) bool {
	return slices.ContainsFunc(models, func(m string) bool {
		return !strings.Contains(m, "cloud")
	})
}

func printModelsAdded(integration *integrationDef, models []string) {
	if integration.Setup != nil {
		if len(models) == 1 {
			fmt.Fprintf(os.Stderr, "Added %s to %s\n", models[0], integration.DisplayName)
		} else {
			fmt.Fprintf(os.Stderr, "Added %d models to %s (default: %s)\n", len(models), integration.DisplayName, models[0])
		}
	}

	if hasLocalModel(models) {
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Coding agents work best with at least 64k context. Either:")
		fmt.Fprintln(os.Stderr, "  - Set the context slider in Ollama app settings")
		fmt.Fprintln(os.Stderr, "  - Run: OLLAMA_CONTEXT_LENGTH=64000 ollama serve")
	}
}

func runIntegration(integrationName, modelName string) error {
	integ, ok := integration(integrationName)
	if !ok {
		return fmt.Errorf("unknown integration: %s", integrationName)
	}

	if err := integ.CheckInstall(); err != nil {
		return err
	}

	if integ.Setup != nil {
		models := []string{modelName}
		if config, err := loadIntegration(integrationName); err == nil && len(config.Models) > 0 {
			models = config.Models
		}
		if err := integ.Setup(models); err != nil {
			return fmt.Errorf("setup failed: %w", err)
		}
	}

	proc := exec.Command(integ.Command, integ.Args(modelName)...)
	proc.Stdin = os.Stdin
	proc.Stdout = os.Stdout
	proc.Stderr = os.Stderr
	proc.Env = os.Environ()
	for _, env := range integ.EnvVars(modelName) {
		proc.Env = append(proc.Env, fmt.Sprintf("%s=%s", env.Name, env.Value))
	}

	fmt.Fprintf(os.Stderr, "\nLaunching %s with %s...\n", integ.DisplayName, modelName)
	return proc.Run()
}

func handleCancelled(err error) (cancelled bool, origErr error) {
	if errors.Is(err, errCancelled) {
		return true, nil
	}
	return false, err
}

// ConfigCmd returns the cobra command for configuring integrations.
func ConfigCmd(checkServerHeartbeat func(cmd *cobra.Command, args []string) error) *cobra.Command {
	var modelFlag string
	var launchFlag bool

	cmd := &cobra.Command{
		Use:   "config [INTEGRATION]",
		Short: "Configure an external integration to use Ollama",
		Long: `Configure an external application to use Ollama models.

Supported integrations:
  claude    Claude Code
  codex     Codex
  droid     Droid
  opencode  OpenCode

Examples:
  ollama config
  ollama config claude
  ollama config droid --launch`,
		Args:    cobra.MaximumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE: func(cmd *cobra.Command, args []string) error {
			var integrationName string
			if len(args) > 0 {
				integrationName = args[0]
			} else {
				var err error
				integrationName, err = selectIntegration()
				cancelled, err := handleCancelled(err)
				if cancelled {
					return nil
				}
				if err != nil {
					return err
				}
			}

			integ, ok := integration(integrationName)
			if !ok {
				return fmt.Errorf("unknown integration: %s", integrationName)
			}

			// If --launch without --model, use saved config if available
			if launchFlag && modelFlag == "" {
				if config, err := loadIntegration(integrationName); err == nil && config.defaultModel() != "" {
					return runIntegration(integrationName, config.defaultModel())
				}
			}

			var models []string
			if modelFlag != "" {
				// When --model is specified, merge with existing models (new model becomes default)
				models = []string{modelFlag}
				if existing, err := loadIntegration(integrationName); err == nil && len(existing.Models) > 0 {
					for _, m := range existing.Models {
						if m != modelFlag {
							models = append(models, m)
						}
					}
				}
			} else {
				var err error
				models, err = selectModels(cmd.Context(), integrationName, "")
				cancelled, err := handleCancelled(err)
				if cancelled {
					return nil
				}
				if err != nil {
					return err
				}
			}

			if integ.Setup != nil {
				paths := integ.configPaths()
				if len(paths) > 0 {
					fmt.Fprintf(os.Stderr, "This will modify your %s configuration:\n", integ.DisplayName)
					for _, p := range paths {
						fmt.Fprintf(os.Stderr, "  %s\n", p)
					}
					fmt.Fprintf(os.Stderr, "Backups will be saved to %s/\n\n", backupDir())

					if ok, _ := confirmPrompt("Proceed?"); !ok {
						return nil
					}
				}
			}

			if err := saveIntegration(integrationName, models); err != nil {
				return fmt.Errorf("failed to save: %w", err)
			}

			if integ.Setup != nil {
				if err := integ.Setup(models); err != nil {
					return fmt.Errorf("setup failed: %w", err)
				}
			}

			printModelsAdded(integ, models)

			if launchFlag {
				return runIntegration(integrationName, models[0])
			}

			if launch, _ := confirmPrompt(fmt.Sprintf("\nLaunch %s now?", integ.DisplayName)); launch {
				return runIntegration(integrationName, models[0])
			}

			fmt.Fprintf(os.Stderr, "Run 'ollama config %s --launch' to start with %s\n", strings.ToLower(integrationName), models[0])
			return nil
		},
	}

	cmd.Flags().StringVar(&modelFlag, "model", "", "Model to use")
	cmd.Flags().BoolVar(&launchFlag, "launch", false, "Launch the integration after configuring")
	return cmd
}
