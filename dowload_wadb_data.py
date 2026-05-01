import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run_id>
run = api.run("massimo-bosetti-liceo-da-vinci/leo-nmt/pd30misx")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("metrics_training.csv")

import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run_id>
run = api.run("massimo-bosetti-liceo-da-vinci/LEO-Translation/gugbcdxx")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("metrics_test.csv")

if __name__ == "__main__":
    main()  