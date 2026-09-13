import requests
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Real-Time Fraud Detection",
    page_icon="💳",
    layout="wide"
)


st.title("💳 Real-Time Fraud Detection System")

st.write(
    "Detect potentially fraudulent credit card transactions "
    "using the trained machine learning model."
)


st.divider()


st.sidebar.header("Prediction Mode")

mode = st.sidebar.radio(
    "Choose a mode:",
    [
        "Manual Entry",
        "CSV Upload"
    ]
)


# ============================================================
# MANUAL ENTRY
# ============================================================

if mode == "Manual Entry":

    st.header("Manual Transaction Prediction")

    st.write(
        "Enter the transaction features below."
    )

    st.subheader("Transaction Information")

    col1, col2 = st.columns(2)

    with col1:

        time_value = st.number_input(
            "Time",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

    with col2:

        amount_value = st.number_input(
            "Amount",
            min_value=0.0,
            value=0.0,
            step=0.01
        )

    st.subheader("Principal Components")

    component_values = {}

    columns = st.columns(4)

    for index in range(1, 29):

        column_position = (index - 1) % 4

        with columns[column_position]:

            component_values[
                f"V{index}"
            ] = st.number_input(
                f"V{index}",
                value=0.0,
                format="%.6f",
                key=f"manual_V{index}"
            )

    st.divider()

    submitted = st.button(
        "Predict Transaction",
        type="primary"
    )

    if submitted:

        transaction = {
            "Time": time_value,
            "Amount": amount_value
        }

        transaction.update(
            component_values
        )

        try:

            response = requests.post(
                "http://localhost:8007/predict",
                json=transaction,
                timeout=60
            )

            if response.status_code == 200:

                result = response.json()

                st.divider()

                st.subheader(
                    "Prediction Result"
                )

                prediction = result["prediction"]
                probability = result["probability"]
                threshold = result["threshold"]

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    if prediction == 1:

                        st.error(
                            "🚨 FRAUDULENT TRANSACTION DETECTED"
                        )

                    else:

                        st.success(
                            "✅ TRANSACTION APPEARS LEGITIMATE"
                        )

                with result_col2:

                    st.metric(
                        "Fraud Probability",
                        f"{probability:.2%}"
                    )

                st.write(
                    f"**Decision threshold:** "
                    f"{threshold:.2%}"
                )

                st.subheader(
                    "Plain-English Explanation"
                )

                st.info(
                    result["explanation"]
                )

                st.subheader(
                    "Most Influential Features"
                )

                feature_rows = []

                for item in result["top_features"]:

                    feature_rows.append(
                        {
                            "Feature": item["feature"],
                            "SHAP Value": item["shap_value"],
                            "Effect": item["direction"]
                        }
                    )

                feature_df = pd.DataFrame(
                    feature_rows
                )

                st.dataframe(
                    feature_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                try:

                    error_detail = response.json()

                except ValueError:

                    error_detail = response.text

                st.error(
                    f"Backend returned HTTP "
                    f"{response.status_code}: "
                    f"{error_detail}"
                )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the FastAPI backend. "
                "Please make sure the backend is running "
                "on port 8007."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The request timed out. Please try again."
            )

        except requests.exceptions.RequestException as error:

            st.error(
                f"Request failed: {error}"
            )


# ============================================================
# CSV BATCH PREDICTION
# ============================================================

else:

    st.header("CSV Batch Prediction")

    st.write(
        "Upload a CSV file containing transaction data."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file
            )

            st.subheader(
                "Uploaded Data"
            )

            st.write(
                f"Rows: {len(df)}"
            )

            st.dataframe(
                df,
                use_container_width=True
            )

            st.divider()

            predict_batch = st.button(
                "Predict All Transactions",
                type="primary"
            )

            if predict_batch:

                results = []

                progress = st.progress(
                    0
                )

                total_rows = len(df)

                for index, row in df.iterrows():

                    transaction = row.to_dict()

                    try:

                        response = requests.post(
                            "http://localhost:8007/predict",
                            json=transaction,
                            timeout=60
                        )

                        if response.status_code == 200:

                            result = response.json()

                            results.append(
                                {
                                    "Row": index + 1,
                                    "Prediction": result[
                                        "prediction"
                                    ],
                                    "Fraud Probability": result[
                                        "probability"
                                    ],
                                    "Message": result[
                                        "message"
                                    ],
                                    "Explanation": result[
                                        "explanation"
                                    ]
                                }
                            )

                        else:

                            results.append(
                                {
                                    "Row": index + 1,
                                    "Prediction": "Error",
                                    "Fraud Probability": None,
                                    "Message": (
                                        f"HTTP "
                                        f"{response.status_code}"
                                    ),
                                    "Explanation": (
                                        "The backend could not "
                                        "process this transaction."
                                    )
                                }
                            )

                    except requests.exceptions.RequestException:

                        results.append(
                            {
                                "Row": index + 1,
                                "Prediction": "Error",
                                "Fraud Probability": None,
                                "Message": (
                                    "Connection error"
                                ),
                                "Explanation": (
                                    "Could not connect to "
                                    "the FastAPI backend."
                                )
                            }
                        )

                    progress.progress(
                        (index + 1) / total_rows
                    )

                progress.empty()

                results_df = pd.DataFrame(
                    results
                )

                st.divider()

                st.subheader(
                    "Batch Prediction Results"
                )

                # Highlight only fraud rows.
                # Normal rows keep Streamlit's
                # default theme colors.
                def highlight_fraud(row):

                    if row["Prediction"] == 1:

                        return [
                            (
                                "background-color: #FFF3CD; "
                                "color: #000000; "
                                "font-weight: bold;"
                            )
                        ] * len(row)

                    return [
                        ""
                    ] * len(row)

                styled_results = results_df.style.apply(
                    highlight_fraud,
                    axis=1
                )

                st.dataframe(
                    styled_results,
                    use_container_width=True,
                    hide_index=True
                )

                fraud_count = (
                    results_df["Prediction"]
                    .eq(1)
                    .sum()
                )

                st.metric(
                    "Flagged Transactions",
                    fraud_count
                )

                st.divider()

                st.subheader(
                    "Transaction Details"
                )

                for result in results:

                    if result["Prediction"] == 1:

                        with st.expander(
                            f"🚨 Transaction "
                            f"{result['Row']} "
                            f"— Fraud Detected"
                        ):

                            st.write(
                                f"**Fraud Probability:** "
                                f"{result['Fraud Probability']:.2%}"
                            )

                            st.write(
                                f"**Message:** "
                                f"{result['Message']}"
                            )

                            st.write(
                                "**Plain-English Explanation:**"
                            )

                            st.info(
                                result["Explanation"]
                            )

                    elif result["Prediction"] == 0:

                        with st.expander(
                            f"✅ Transaction "
                            f"{result['Row']} "
                            f"— Appears Legitimate"
                        ):

                            st.write(
                                f"**Fraud Probability:** "
                                f"{result['Fraud Probability']:.2%}"
                            )

                            st.write(
                                f"**Message:** "
                                f"{result['Message']}"
                            )

                            st.write(
                                "**Plain-English Explanation:**"
                            )

                            st.info(
                                result["Explanation"]
                            )

        except Exception as error:

            st.error(
                f"Could not read the CSV file: "
                f"{error}"
            )