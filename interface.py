import streamlit as st
import importlib
import src.utils
import src.models
import src.counterfactual
import src.train

importlib.reload(src.utils)
importlib.reload(src.models)
importlib.reload(src.counterfactual)
importlib.reload(src.train)

from src.utils import (
    load_data,
    load_model,
    clean_instance,
    DatasetMetadata,
    transform_onehot_inverse,
)
from src.counterfactual import unscale_instance, scale_instance, newton_op
from src.train import main as main_train
import torch
import pandas as pd
import math

# str to sympy
from src.models import LogisticModel
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def write_sample_to_state(sample_row: pd.Series, cols, changeable):
    """
    Copy every column in `cols` from `sample_row` into st.session_state.
    """
    for col in cols:
        st.session_state[f"input_{col}"] = sample_row[col]
        # If you also want to set the checkbox state:
        if col in changeable:
            st.session_state[f"check_{col}"] = bool(changeable[col])


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(samples: pd.DataFrame, model: torch.nn.Module, metadata: DatasetMetadata):
    st.title("Bank Loan Application Simulator")
    st.write(
        "Welcome! Please fill out the form below to see if you're eligible for a loan. "
        "If your application is denied, we'll show recommended actions."
    )

    # Read dataset for reference
    raw_df = pd.read_csv(metadata.path)
    raw_df = raw_df.drop(columns=[metadata.target_column, metadata.id_column], errors="ignore")
    cols = raw_df.columns
    samples[metadata.cols_for_scaler_names] = metadata.scaler.inverse_transform(
        samples[metadata.cols_for_scaler_names]
    )
    samples = transform_onehot_inverse(samples, metadata)

    inputs = {}
    changeable = {}
    weights_slider = {}

    # ------------------------------------------------------------------
    # 1️⃣  Prefill buttons
    # ------------------------------------------------------------------
    col_pos, col_neg = st.columns(2)
    with col_pos:
        if st.button("🔵 Load positive sample"):
            write_sample_to_state(samples.iloc[metadata.good_class], cols, changeable={})
            st.rerun()
    with col_neg:
        if st.button("🔴 Load negative sample"):
            write_sample_to_state(samples.iloc[1 - metadata.good_class], cols, changeable={})
            st.rerun()

    # ------------------------------------------------------------------
    # 2️⃣  Build the form
    # ------------------------------------------------------------------
    with st.form(key="loan_application_form"):
        for col in cols:
            # Use any value already stored in session_state as the default
            default_val = st.session_state.get(f"input_{col}", None)

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{col}**")
                if raw_df[col].dtype == "object":
                    options = raw_df[col].unique()
                    # Convert default value into its position in `options`
                    index = (
                        list(options).index(default_val)
                        if default_val in options
                        else 0
                    )
                    inputs[col] = st.selectbox(
                        f"Select {col}",
                        options=options,
                        index=index,
                        key=f"input_{col}",
                    )
                elif raw_df[col].dtype == "int64":
                    inputs[col] = st.number_input(
                        f"Enter {col}",
                        min_value=int(raw_df[col].min()),
                        max_value=int(raw_df[col].max()),
                        value=int(default_val or raw_df[col].mean()),
                        step=1,
                        key=f"input_{col}",
                    )
                else:  # float32
                    inputs[col] = st.number_input(
                        f"Enter {col}",
                        min_value=float(raw_df[col].min()),
                        max_value=float(raw_df[col].max()),
                        value=float(default_val or raw_df[col].mean()),
                        step=0.1,
                        key=f"input_{col}",
                    )
            with col2:
                if col in metadata.changeable_col_names:
                    changeable[col] = st.checkbox(
                        "Changeable",
                        value=st.session_state.get(f"check_{col}", True),
                        key=f"check_{col}",
                    )
            with col3:
                if col in metadata.changeable_col_names:
                    log_val = st.slider(
                        "Weight (log10)",
                        min_value=0.1,
                        max_value=10.0,
                        # value=math.log10( st.session_state.get(f"weight_{col}", 1.0) ),
                        value=1.0,
                        step=0.05,          # res. ≈ 10^(0.05) ≈ 1.12
                        key=f"logweight_{col}",
                        help="-1 → easy to change | +1 → hard to change",
                    )
                    # weight = 10 ** log_val
                    weights_slider[col] = log_val

                    # Guarda el valor real (lineal) en session_state
                    st.session_state[f"weight_{col}"] = log_val
                    st.caption(f"{log_val:.2f}")
        # Once the user clicks the button, we collect the inputs and run inference
        if st.form_submit_button("Submit Application"):
            # Format data for your model (reshape or convert to DataFrame as needed)
            # Example: collecting only 3 fields
            # user_features = np.array([inputs[col] for col in cols]).reshape(1, -1)

            # Convert to DataFrame if needed
            person = clean_instance(inputs, metadata)
            # weights are the columns that are changeable
            list_weights = [
                weights_slider[col] if changeable.get(col, False) else 0 for col in metadata.columns
            ]
            weights = torch.tensor(list_weights, dtype=torch.float32).to(device)
            # print(f"weights: {weights}")
            result = model(person.unsqueeze(0))[0][metadata.good_class].item() > metadata.threshold

            # 2) Display results
            if result:
                st.success("Congratulations! Your loan is approved.")
            else:
                st.error("Your loan was denied.")
                st.write("Here's what we recommend so you can qualify next time:")
                                
                person_new, _ = newton_op(
                    model,
                    person,
                    metadata,
                    weights,
                    delta_threshold=0.2,
                    reg_int=True,
                    reg_clamp=True,
                    print_=True,
                )

                person_unscaled = unscale_instance(person, metadata)
                person_new_unscaled = unscale_instance(person_new, metadata)

                # 3) Call your recourse logic. For example:
                # recourse_instructions = generate_recourse(user_features)
                # st.write(recourse_instructions)

                # Simulated recourse for demonstration:
                changes = torch.round(person_new_unscaled, decimals=2) - torch.round(
                    person_unscaled, decimals=2
                )
                for i, col in enumerate(metadata.columns):
                    if changes[i] > 0:
                        st.write(
                            f"- Increase {col} from {person_unscaled[i].item():.2f} to {person_new_unscaled[i].item():.2f} (+{(person_new_unscaled[i].item() - person_unscaled[i].item()) / person_unscaled[i].item() * 100:.2f}%)"
                        )
                    elif changes[i] < 0:
                        st.write(
                            f"- Decrease {col} from {person_unscaled[i].item():.2f} to {person_new_unscaled[i].item():.2f} ({(person_new_unscaled[i].item() - person_unscaled[i].item()) / person_unscaled[i].item() * 100:.2f}%)"
                        )

                st.write("These adjustments should help get an approved decision.")


if __name__ == "__main__":
    filename = "data/Loan_default.csv"
    model_name = "model_small"
    # model = main_train(filename, model_name)
    model = load_model(model_name).to(device)
    # load a pickle file
    # model = joblib.load(open("models/german_model.joblib", "rb"))

    person: torch.Tensor
    metadata: DatasetMetadata
    df = pd.read_csv(filename)
    samples, metadata = load_data(filename, get_sample=True)

    main(samples, model, metadata)
