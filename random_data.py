import math
import streamlit as st
import streamlit_option_menu
from streamlit_option_menu import option_menu
import numpy as np

def prediction_from_user_data(df, model, keys, random=False):
    col1, col2, col3 = st.columns(3)
    df.columns = df.columns.str.lower()
    
    selected_features = ['sex', 'r_lfa', 'pns', 'imped_lat', 'bacq_withdrawal_coping_sum', 'bas_drive_sum', 'bas_entertainment_sum', 'hq_logical_orientation_right', 'hq_type_of_consciousness_right', 'hq_fear_level_and_sensitivity_left', 'hq_fear_level_and_sensitivity_right', 'hq_pair_bonding_and_spouse_dominance_style_left', 'hq_pair_bonding_and_spouse_dominance_style_right', 'wapi_verbal_left', 'rsq_anxious_attachment']
    
    
    if 'random_row' not in st.session_state:
        st.session_state.random_row = df.sample(n=1)
    
    random_row = st.session_state.random_row
    features = random_row[selected_features].values
    
    if "index" not in st.session_state:
        st.session_state["index"] = {random_row.index[0]}
    if random:
        c1, c2 = st.columns(2)
        with c1: st.warning(f"Serial: {list(st.session_state['index'])[0] + 2}")
        with c2: st.warning(f"Anonymous patient data of serial {list(st.session_state['index'])[0] + 2} from the dataset.")
    
    print(random_row[selected_features + ['group']])
    
    columns = [
    "sex",
    "r_lfa",
    "pns",
    "imped_lat",
    "bacq_withdrawal_coping_sum",
    "bas_drive_sum",
    "bas_entertainment_sum",
    "hq_logical_orientation_right",
    "hq_type_of_consciousness_right",
    "hq_fear_level_and_sensitivity_left",
    "hq_fear_level_and_sensitivity_right",
    "hq_pair_bonding_and_spouse_dominance_style_left",
    "hq_pair_bonding_and_spouse_dominance_style_right",
    "wapi_verbal_left",
    "rsq_anxious_attachment"
    ]
    
    values = {col: random_row[col].fillna(0.0).values[0] for col in columns}
    
    if not random:
        values = {col: 0.0 for col in selected_features}
    
    sex_value = values["sex"] if random else 1.0
    r_lfa_value = values["r_lfa"]
    pns_value = values["pns"]
    imped_lat_value = values["imped_lat"]
    bacq_withdrawal_coping_sum_value = values["bacq_withdrawal_coping_sum"]
    bas_drive_sum_value = values["bas_drive_sum"]
    bas_entertainment_sum_value = values["bas_entertainment_sum"]
    hq_logical_orientation_right_value = values["hq_logical_orientation_right"]
    hq_type_of_consciousness_right_value = values["hq_type_of_consciousness_right"]
    hq_fear_level_and_sensitivity_left_value = values["hq_fear_level_and_sensitivity_left"]
    hq_fear_level_and_sensitivity_right_value = values["hq_fear_level_and_sensitivity_right"]
    hq_pair_bonding_and_spouse_dominance_style_left_value = values["hq_pair_bonding_and_spouse_dominance_style_left"]
    hq_pair_bonding_and_spouse_dominance_style_right_value = values["hq_pair_bonding_and_spouse_dominance_style_right"]
    wapi_verbal_left_value = values["wapi_verbal_left"]
    rsq_anxious_attachment_value = values["rsq_anxious_attachment"]
    
    if random:
        for col in selected_features:
            if col not in st.session_state:
                st.session_state[col] = values[col]
                
    if random:
        sex_value = st.session_state["sex"]
        r_lfa_value = st.session_state["r_lfa"]
        pns_value = st.session_state["pns"]
        imped_lat_value = st.session_state["imped_lat"]
        bacq_withdrawal_coping_sum_value = st.session_state["bacq_withdrawal_coping_sum"]
        bas_drive_sum_value = st.session_state["bas_drive_sum"]
        bas_entertainment_sum_value = st.session_state["bas_entertainment_sum"]
        hq_logical_orientation_right_value = st.session_state["hq_logical_orientation_right"]
        hq_type_of_consciousness_right_value = st.session_state["hq_type_of_consciousness_right"]
        hq_fear_level_and_sensitivity_left_value = st.session_state["hq_fear_level_and_sensitivity_left"]
        hq_fear_level_and_sensitivity_right_value = st.session_state["hq_fear_level_and_sensitivity_right"]
        hq_pair_bonding_and_spouse_dominance_style_left_value = st.session_state["hq_pair_bonding_and_spouse_dominance_style_left"]
        hq_pair_bonding_and_spouse_dominance_style_right_value = st.session_state["hq_pair_bonding_and_spouse_dominance_style_right"]
        wapi_verbal_left_value = st.session_state["wapi_verbal_left"]
        rsq_anxious_attachment_value = st.session_state["rsq_anxious_attachment"]
    
    
    with col1:
        # sex = st.number_input('Sex (1 or 2)', key=f"{keys}_input_1", min_value=1.0, max_value=2.0, value=sex_value, step=1.0)
        sex = st.radio('Sex', options=['Male', 'Female'], key=f"{keys}_input_1", index=int(sex_value - 1))
        if not random: sex_value = 1 if sex == 'Male' else 2
        # st.write(f"Selected sex value: {sex_value}")
        
        imped_lat = st.number_input('Impedance Latency', key=f"{keys}_input_2", min_value=-5.0, value=imped_lat_value, format="%.10f")
        bas_entertainment_sum = st.number_input('BAS Entertainment Sum', key=f"{keys}_input_3", min_value=-50.0, max_value=50.0,   value=bas_entertainment_sum_value)
        hq_fear_level_and_sensitivity_left = st.number_input('HQ Fear Level Left', key=f"{keys}_input_4", min_value=-10.0, max_value=10.0,     value=hq_fear_level_and_sensitivity_left_value)
        hq_pair_bonding_and_spouse_dominance_style_right = st.number_input('HQ Pair Bonding Right', key=f"{keys}_input_5", min_value=-10.0,    max_value=10.0, value=hq_pair_bonding_and_spouse_dominance_style_right_value)
    
    with col2:
        r_lfa = st.number_input('r_LFA', key=f"{keys}_input_6", min_value=-5.0, max_value=5.0, value=r_lfa_value, format="%.10f")
        bacq_withdrawal_coping_sum = st.number_input('BACQ Withdrawal Coping Sum', key=f"{keys}_input_7", min_value=-50.0, max_value=50.0,     value=bacq_withdrawal_coping_sum_value)
        hq_logical_orientation_right = st.number_input('HQ Logical Orientation Right', key=f"{keys}_input_8", min_value=-5.0,  value=hq_logical_orientation_right_value)
        hq_fear_level_and_sensitivity_right = st.number_input('HQ Fear Level Right', key=f"{keys}_input_9", min_value=-10.0, max_value=10.0,   value=hq_fear_level_and_sensitivity_right_value)
        wapi_verbal_left = st.number_input('WAPI Verbal Left', key=f"{keys}_input_10", min_value=-20.0, max_value=20.0,     value=wapi_verbal_left_value)
        
    
    with col3:
        pns = st.number_input('PNS', key=f"{keys}_input_11", min_value=-10.0, max_value=10.0, value=pns_value)
        bas_drive_sum = st.number_input('BAS Drive Sum', key=f"{keys}_input_12", min_value=-50.0, max_value=50.0, value=bas_drive_sum_value)
        hq_type_of_consciousness_right = st.number_input('HQ Type of Consciousness Right', key=f"{keys}_input_13", min_value=-10.0,     max_value=10.0, value=hq_type_of_consciousness_right_value)
        hq_pair_bonding_and_spouse_dominance_style_left = st.number_input('HQ Pair Bonding Left', key=f"{keys}_input_14", min_value=-10.0,  max_value=10.0, value=hq_pair_bonding_and_spouse_dominance_style_left_value)
        rsq_anxious_attachment = st.number_input('RSQ Anxious Attachment', key=f"{keys}_input_15", min_value=-50.0, max_value=50.0,     value=rsq_anxious_attachment_value)
    
    
    custom_data = np.array([
        sex_value, r_lfa, pns, imped_lat, bacq_withdrawal_coping_sum,
        bas_drive_sum, bas_entertainment_sum, hq_logical_orientation_right,
        hq_type_of_consciousness_right, hq_fear_level_and_sensitivity_left,
        hq_fear_level_and_sensitivity_right, hq_pair_bonding_and_spouse_dominance_style_left,
        hq_pair_bonding_and_spouse_dominance_style_right, wapi_verbal_left, rsq_anxious_attachment
    ]).reshape(1, -1)
    
    
    c1, c2 = st.columns([1, 6])
    
    class_name = None

    
    with c1:
        if st.button('Run Prediction', key=f"{keys}_random_data_button"):
            predictions = model.predict(custom_data)
            predicted_class = np.argmax(predictions, axis=1)
            if predicted_class == 0:
                class_name = "Healthy Control"
            elif predicted_class == 1:
                class_name = "Atopic Dermatitis"
            else:
                class_name = "Psoriasis"
                
    # if class_name: st.markdown(f"""- Predicted Class: **`{predicted_class[0]}`**
    #                            - Predicted Class Name: **`{class_name}`**
    #                            - Original Class Name: **`{random_row.group.to_list()[0]}`**
    #                            - Confidence Level: **`{max(predictions[0]) * 100:.2f}%`**
    #                            """)
    if class_name:
        groups = {0: "Healthy Control", 1: "Atopic Dermatitis", 2: "Psoriasis"}
        model_predicted_class = predicted_class[0]
        model_predicted_class_name = class_name
        original_class = random_row.group.to_list()[0]
        
        if not math.isnan(original_class):
            original_class = int(original_class) 
        else:
            original_class = 0
        original_class_name = groups[original_class]
        confidence_level = f"{max(predictions[0]) * 100:.2f}"
        
        matched = False
        
        if model_predicted_class == original_class:
            matched = True
            
        text_color = "white"
        
        if matched:
            bg_color = "#80d47c"
            add = "and also"
            message = "Bingo!"
            text_color = "#3e4a3c"
        else:
            bg_color = "red"
            add = "but"
            message = "Wrong Classification!"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""<div style="padding: 10px; border: 1px solid #d1d1d1; border-radius: 5px;">
                            <strong>Predicted Psychodermatological Risk:</strong> <br> <code style='font-weight: bold;'>{model_predicted_class}: {model_predicted_class_name}</code>
                            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""<div style="padding: 10px; border: 1px solid #d1d1d1; border-radius: 5px;">
                            <strong>AI Confidence Level:</strong><br> <code style='font-weight: bold;'>{confidence_level}%</code>
                            </div>""", unsafe_allow_html=True)
            
        if random: 
            with col3: st.markdown(f"""<div style="padding: 10px; border: 1px solid #d1d1d1; border-radius: 5px;">
                            <strong>Original Psychodermatological Risk:</strong><br> <code style='font-weight: bold;'>{original_class}: {original_class_name}</code>
                            </div>""", unsafe_allow_html=True)
        
        if random: st.markdown(f"""<div style="padding: 10px; margin-top:10px; border: 1px solid #d1d1d1; border-radius: 5px; background-color: {bg_color}; color: {text_color};">
                    <strong>{message}</strong><br> <div style=''>The original Psychodermatological Risk was <strong>{original_class}: {original_class_name}</strong> {add} the model predicted 
                    <strong>{model_predicted_class}: {model_predicted_class_name}</strong>.</div>
                    </div>""", unsafe_allow_html=True)
        
        if random: st.warning("""
                   This "Get Random Data from Dataset" is basically a tab for AI model to check its accuracy classifying the groups. In this tab, a random patient’s real data will be fetched from the dataset that we have. Then, if you run the prediction, you can see the AI model is correct or not.
                   """)

    
    with c2:
        if st.button("Refresh", key=f"{keys}_refresh"):
            for col in selected_features:
                del st.session_state[col]
            
            del st.session_state["random_row"]
            del st.session_state["index"]
            
            try:
                st.rerun()
            except Exception as e:
                st.experimental_rerun()
            
            # st.rerun()




