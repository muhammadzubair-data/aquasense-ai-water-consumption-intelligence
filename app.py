import streamlit as st
import pandas as pd
import plotly.express as px
from src.analytics import generate_data, validate_data, add_anomalies, leak_signals, forecast_meter

st.set_page_config(page_title='AquaSense AI | ZubairLabs', page_icon='💧', layout='wide')
st.markdown('''<style>.block-container{padding-top:2rem}div[data-testid="stMetric"]{background:#0e2432;border:1px solid #1d6576;padding:16px;border-radius:14px}div[data-testid="stMetric"] *{color:#e5fbff!important}</style>''', unsafe_allow_html=True)
st.title('💧 AquaSense AI')
st.caption('Smart Water Consumption • Anomaly Signals • Usage Forecasting | ZubairLabs')
st.info('Demonstration only: synthetic data, statistical anomaly signals, and heuristic leak indicators are not verified leak diagnoses.')
@st.cache_data
def demo(): return generate_data()
upload = st.sidebar.file_uploader('Upload water-meter CSV', type='csv')
try:
    df = validate_data(pd.read_csv(upload) if upload else demo())
except Exception as e:
    st.error(f'Dataset error: {e}')
    st.stop()
st.sidebar.download_button('Download sample CSV', demo().to_csv(index=False), 'aquasense_sample.csv', 'text/csv')
meter = st.sidebar.selectbox('Select meter', sorted(df.meter_id.unique()))
view = df[df.meter_id == meter].copy()
tabs = st.tabs(['Command Center', 'Anomaly Detection', 'Leak Signals', 'Demand Forecast', 'Data Explorer', 'Methodology'])
with tabs[0]:
    a,b,c,d = st.columns(4)
    a.metric('Meters', df.meter_id.nunique())
    b.metric('Total liters', f'{df.liters.sum():,.0f}')
    c.metric('Mean liters / reading', f'{df.liters.mean():.1f}')
    d.metric('Readings', f'{len(df):,}')
    daily = df.assign(date=df.timestamp.dt.date).groupby('date',as_index=False).liters.sum()
    st.plotly_chart(px.line(daily,x='date',y='liters',title='Daily water consumption'),use_container_width=True)
    st.plotly_chart(px.bar(df.groupby('meter_id',as_index=False).liters.sum(),x='meter_id',y='liters',title='Consumption by meter'),use_container_width=True)
with tabs[1]:
    if len(view) < 20: st.warning('At least 20 readings required')
    else:
        result = add_anomalies(view)
        st.metric('Anomalous readings', int(result.anomaly.sum()))
        fig = px.scatter(result,x='timestamp',y='liters',color='anomaly',title='Statistical anomalies (not confirmed leaks)')
        st.plotly_chart(fig,use_container_width=True)
        st.download_button('Download anomaly results',result.to_csv(index=False),'anomaly_signals.csv','text/csv')
with tabs[2]:
    threshold = st.slider('Sustained-use threshold multiplier',1.1,2.5,1.35,0.05)
    window = st.slider('Rolling readings',6,48,12)
    result = leak_signals(view,window=window,threshold=threshold)
    st.metric('Elevated-use readings',int(result.leak_signal.sum()))
    st.plotly_chart(px.scatter(result,x='timestamp',y='liters',color='leak_signal',title='Heuristic sustained-use signals'),use_container_width=True)
    st.download_button('Download leak signals',result.to_csv(index=False),'elevated_use_signals.csv','text/csv')
with tabs[3]:
    horizon = st.slider('Forecast hours',6,72,24,6)
    try:
        with st.spinner('Training forecast model...'):
            future, metrics = forecast_meter(view,meter,horizon)
        c1,c2=st.columns(2)
        c1.metric('Holdout MAE (liters)',f'{metrics["MAE"]:.2f}')
        c2.metric('Holdout RMSE (liters)',f'{metrics["RMSE"]:.2f}')
        plot = pd.concat([view[['timestamp','liters']].tail(96).assign(series='Observed'),future.rename(columns={'forecast_liters':'liters'}).assign(series='Forecast')])
        st.plotly_chart(px.line(plot,x='timestamp',y='liters',color='series',title='Observed and recursive hourly forecast'),use_container_width=True)
        st.download_button('Download forecast',future.to_csv(index=False),'water_forecast.csv','text/csv')
    except ValueError as e: st.warning(str(e))
with tabs[4]:
    st.dataframe(view,use_container_width=True)
    st.download_button('Download selected meter',view.to_csv(index=False),'meter_readings.csv','text/csv')
with tabs[5]:
    st.markdown('''### Methods and limitations
- **Anomaly detection:** Isolation Forest using consumption, hour and weekday; contamination is a modeling assumption.
- **Elevated-use signals:** trailing rolling mean compared against a historical expanding median; does not establish a leak.
- **Forecasting:** Random Forest with lag-1, lag-24, trailing 24-hour average and calendar features. Chronological 80/20 holdout, then refit and recursive forecast.
- **Data:** synthetic demonstration readings. No real-world validation, leak confirmation, water quality assessment, or production readiness claimed.
- **CSV schema:** `timestamp,meter_id,liters`; timestamps must be parseable and readings nonnegative.
''')
st.caption('© ZubairLabs • github.com/muhammadzubair-data')
