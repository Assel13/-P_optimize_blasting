import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Загрузка модели, скейлера и списка признаков
try:
    model = joblib.load('C:/Users/HP/OneDrive/Documents/StreamlitProject/model.pkl')
    scaler = joblib.load('C:/Users/HP/OneDrive/Documents/StreamlitProject/scaler.pkl')
    feature_names = joblib.load('C:/Users/HP/OneDrive/Documents/StreamlitProject/feature_names.pkl')
except Exception as e:
    st.error(f"Ошибка при загрузке модели, скейлера или признаков: {str(e)}")
    st.stop()


# Функция предсказания P20
def predict_p20(powder_factor, burden, spacing, ucs):
    try:
        burden_spacing_ratio = burden / spacing if spacing != 0 else 0
        ucs_log = np.log1p(ucs)

        # Создаём DataFrame в правильном порядке
        correct_order = ['NUM_POWDER_FACTOR', 'NUM_BURDEN', 'NUM_SPACING', 'UCS_log']

        input_data = pd.DataFrame({
            'NUM_POWDER_FACTOR': [powder_factor],
            'NUM_BURDEN': [burden],
            'NUM_SPACING': [spacing],
            'UCS_log': [ucs_log]
        })[correct_order]

        # ✅ Исправленный отступ:
        print("Перед transform():", input_data)

        # Масштабируем входные данные
        input_scaled = scaler.transform(input_data)

        # Прогноз
        prediction = model.predict(input_scaled)
        return prediction[0]

    except Exception as e:
        st.error(f"Ошибка при предсказании: {str(e)}")
        return None
    
# Интерфейс Streamlit
st.title("Оптимизация взрывных работ с ML")
st.write("Прогнозирование размера частиц P20 на основе параметров взрывных работ.")

# Ввод параметров пользователем
powder_factor = st.slider("NUM_POWDER_FACTOR (кг/м³)", 0.7, 1.1, 0.85)
burden = st.slider("NUM_BURDEN (м)", 5.0, 6.0, 5.5)
spacing = st.slider("NUM_SPACING (м)", 5.0, 6.5, 5.5)
ucs = st.slider("UCS (МПа)", 30.0, 100.0, 50.0)

# 🔍 Вывод параметров перед передачей в `predict_p20()`
st.write(f"Перед отправкой в predict_p20: Powder Factor = {powder_factor}, Burden = {burden}, Spacing = {spacing}, UCS = {ucs}")

# Кнопка для запуска предсказания
if st.button("Прогноз P20"):
    p20 = predict_p20(powder_factor, burden, spacing, ucs)
    if p20 is not None:
        st.write(f"Прогноз P20 (мм): {p20:.2f}")
        if 30 <= p20 <= 50:
            st.success("P20 в целевом диапазоне (30–50 мм)!")
        else:
            st.warning("P20 вне целевого диапазона (30–50 мм). Проверьте параметры или модель.")
    else:
        st.error("Не удалось выполнить прогноз. Проверьте введенные данные.")

# Отображение текущих параметров
st.write("Текущие параметры:")
st.write(f"Powder Factor: {powder_factor} кг/м³")
st.write(f"Burden: {burden} м")
st.write(f"Spacing: {spacing} м")
st.write(f"UCS: {ucs} МПа")
st.write(f"Burden/Spacing Ratio: {burden/spacing:.2f}")