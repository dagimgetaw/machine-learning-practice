import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import dataclasses
from sklearn.preprocessing import LabelEncoder


loan_dataset = pd.read_csv('loan_approval_dataset.csv')

feature_mean = loan_dataset.mean(numeric_only=True)
feature_std = loan_dataset.std(numeric_only=True)
numeric_feature = loan_dataset.select_dtypes('number').columns
normalized_dataset = (loan_dataset[numeric_feature] - feature_mean) / feature_std

normalized_dataset.head()

label_encoder = LabelEncoder()

loan_dataset['education'] = label_encoder.fit_transform(loan_dataset['education'])
loan_dataset['self_employed'] = label_encoder.fit_transform(loan_dataset['self_employed'])
loan_dataset['loan_status'] = label_encoder.fit_transform(loan_dataset['loan_status'])

normalized_dataset['education'] = loan_dataset['education']
normalized_dataset['self_employed'] = loan_dataset['self_employed']
normalized_dataset['loan_status'] = loan_dataset['loan_status']

keras.utils.set_random_seed(42)

number_sample = len(normalized_dataset)
index_80 = round(number_sample * 0.8)
index_90 = index_80 + round(number_sample * 0.1)

shuffled_data = normalized_dataset.sample(frac=1, random_state=100)
train_data = shuffled_data.iloc[0:index_80]
validation_data = shuffled_data.iloc[index_80:index_90]
test_data = shuffled_data.iloc[index_90:]

droped_columns = ['loan_id', 'loan_status']

train_features = train_data.drop(columns=droped_columns)
train_labels = train_data['loan_status'].to_numpy()

validation_features = validation_data.drop(columns=droped_columns)
validation_labels = validation_data['loan_status'].to_numpy()

test_features = test_data.drop(columns=droped_columns)
test_labels = test_data['loan_status'].to_numpy()


correlation_matrix = loan_dataset.corr(numeric_only=True)

correlation_with_label = correlation_matrix['loan_status'].abs().sort_values(ascending=False)

input_features = [
    'cibil_score ',
    'loan_term',
]

@dataclasses.dataclass()
class ExperimentSettings:
    learning_rate: float
    number_epochs: int
    batch_size: int
    classification_threshold: float
    input_features: list[str]


@dataclasses.dataclass()
class Experiment:
    name: str
    settings: ExperimentSettings
    model: keras.Model
    epochs: np.ndarray
    metrics_history: keras.callbacks.History

    def get_final_metric_value(self, metric_name: str) -> float:
        if metric_name not in self.metrics_history:
            raise ValueError(
                f'Unknown metric {metric_name}: available metrics are'
                f' {list(self.metrics_history.columns)}'
            )
        return self.metrics_history[metric_name].iloc[-1]


def create_model(settings: ExperimentSettings, metrics: list[keras.metrics.Metric]) -> keras.Model:
    model_inputs = [keras.Input(name=feature, shape=(1,)) for feature in settings.input_features]

    concatenated_inputs = keras.layers.Concatenate()(model_inputs)
    dense = keras.layers.Dense(units=1, input_shape=(1,), name='dense_layer', activation=keras.activations.sigmoid)
    model_output = dense(concatenated_inputs)
    model = keras.Model(inputs=model_inputs, outputs=model_output)
    model.compile(
        optimizer=keras.optimizers.RMSprop(
            settings.learning_rate
        ),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )
    return model

def train_model(
        experiment_name: str,
        model: keras.Model,
        dataset: pd.DataFrame,
        labels: np.ndarray,
        settings: ExperimentSettings,
) -> Experiment:
    features = {
        feature_name: np.array(dataset[feature_name])
        for feature_name in settings.input_features
    }

    history = model.fit(
        x=features,
        y=labels,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs,
    )

    return Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )

def plot_experiment_metrics(experiment: Experiment, metrics: list[str]):
    plt.figure(figsize=(6, 3))

    for metric in metrics:
        plt.plot(
            experiment.epochs, experiment.metrics_history[metric], label=metric
        )

    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.grid()
    plt.legend()

settings = ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.35,
    input_features=input_features,
)

metrics = [
    keras.metrics.BinaryAccuracy(
        name='accuracy', threshold=settings.classification_threshold
    ),
    keras.metrics.Precision(
        name='precision', thresholds=settings.classification_threshold
    ),
    keras.metrics.Recall(
        name='recall', thresholds=settings.classification_threshold
    ),
    keras.metrics.AUC(num_thresholds=100, name='auc'),
]

# Establish the model's topography.
model = create_model(settings, metrics)

# Train the model on the training set.
experiment = train_model(
    'baseline', model, train_features, train_labels, settings
)
