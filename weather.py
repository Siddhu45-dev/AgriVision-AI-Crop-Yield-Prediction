import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# AGRIVISION AI - NATURE GLASSMORPHISM WEATHER INTELLIGENCE
# =========================================================

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# तुमची निवडलेली Aerial Farm बॅकग्राउंड इमेज
BACKGROUND_URL = (
    "https://images.unsplash.com/photo-1586771107445-d3ca888129ff"
    "?q=80&w=1920&auto=format&fit=crop"
)

WEATHER_CODES = {
    0: ("Clear Sky", "☀️"),
    1: ("Mainly Clear", "🌤️"),
    2: ("Partly Cloudy", "⛅"),
    3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"),
    48: ("Rime Fog", "🌫️"),
    51: ("Light Drizzle", "🌦️"),
    53: ("Moderate Drizzle", "🌦️"),
    55: ("Heavy Drizzle", "🌧️"),
    61: ("Slight Rain", "🌦️"),
    63: ("Moderate Rain", "🌧️"),
    65: ("Heavy Rain", "🌧️"),
    71: ("Slight Snow", "🌨️"),
    73: ("Moderate Snow", "🌨️"),
    75: ("Heavy Snow", "❄️"),
    80: ("Rain Showers", "🌦️"),
    81: ("Moderate Showers", "🌧️"),
    82: ("Heavy Showers", "⛈️"),
    95: ("Thunderstorm", "⛈️"),
    96: ("Thunderstorm with Hail", "⛈️"),
    99: ("Heavy Thunderstorm", "⛈️"),
}


@st.cache_data(ttl=1800, show_spinner=False)
def get_coordinates(city):
    response = requests.get(
        GEOCODING_URL,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=15,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        return None
    place = results[0]
    return {
        "name": place["name"],
        "country": place.get("country", ""),
        "state": place.get("admin1", ""),
        "latitude": place["latitude"],
        "longitude": place["longitude"],
    }


@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(latitude, longitude):
    response = requests.get(
        WEATHER_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "apparent_temperature,"
                "precipitation,"
                "weather_code,"
                "wind_speed_10m,"
                "surface_pressure"
            ),
            "daily": (
                "weather_code,"
                "temperature_2m_max,"
                "temperature_2m_min,"
                "precipitation_sum,"
                "precipitation_probability_max,"
                "wind_speed_10m_max,"
                "uv_index_max,"
                "et0_fao_evapotranspiration,"
                "sunrise,"
                "sunset"
            ),
            "forecast_days": 7,
            "timezone": "auto",
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def load_glass_design():
    st.markdown(
        f"""
        <style>
        /* Force Aerial Farm Background across Streamlit containers */
        [data-testid="stAppViewContainer"], .stApp {{
            background: linear-gradient(rgba(6, 24, 15, 0.52), rgba(5, 18, 12, 0.68)),
                        url("{BACKGROUND_URL}") !important;
            background-size: cover !important;
            background-position: center center !important;
            background-attachment: fixed !important;
        }}

        [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        /* Hero Banner */
        .weather-hero {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.22), rgba(6, 78, 59, 0.45));
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid rgba(72, 245, 181, 0.35);
            border-radius: 24px;
            padding: 24px 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 35px rgba(0,0,0,0.28);
        }}

        .weather-hero h1 {{
            color: #ffffff !important;
            font-size: 32px;
            font-weight: 800;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .weather-hero p {{
            color: #d1fae5 !important;
            font-size: 14px;
            margin: 6px 0 0 0;
            line-height: 1.6;
        }}

        .weather-pill {{
            display: inline-block;
            margin-top: 10px;
            padding: 4px 12px;
            border-radius: 20px;
            background: rgba(72, 245, 181, 0.18);
            border: 1px solid rgba(72, 245, 181, 0.35);
            color: #48f5b5;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}

        /* Frosted Glass Search Form (White background काढले आहे) */
        [data-testid="stForm"] {{
            background: rgba(255, 255, 255, 0.08) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(255, 255, 255, 0.18) !important;
            border-radius: 20px !important;
            padding: 20px 24px !important;
            box-shadow: 0 8px 30px rgba(0,0,0,0.25) !important;
            margin-bottom: 22px;
        }}

        /* Input styling */
        div[data-baseweb="input"] > div {{
            background-color: rgba(10, 28, 20, 0.65) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
        }}
        div[data-baseweb="input"] input {{
            color: #ffffff !important;
            font-size: 15px !important;
        }}
        div[data-baseweb="input"]:focus-within > div {{
            border-color: #48f5b5 !important;
            box-shadow: 0 0 14px rgba(72, 245, 181, 0.35) !important;
        }}
        label {{
            color: #48f5b5 !important;
            font-weight: 700 !important;
            font-size: 13px !important;
        }}

        /* Submit Button */
        div[data-testid="stFormSubmitButton"] > button {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
            color: white !important;
            font-weight: 800 !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            border-radius: 12px !important;
            padding: 11px 24px !important;
            box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4) !important;
            transition: all 0.25s ease !important;
        }}
        div[data-testid="stFormSubmitButton"] > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 22px rgba(72, 245, 181, 0.6) !important;
        }}

        /* Current Weather Showcase */
        .current-weather {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(7, 40, 30, 0.60));
            backdrop-filter: blur(20px);
            border: 1px solid rgba(72, 245, 181, 0.4);
            border-radius: 24px;
            padding: 26px 30px;
            margin: 15px 0 22px 0;
            box-shadow: 0 10px 35px rgba(0,0,0,0.3);
        }}

        .current-temperature {{
            font-size: 58px;
            font-weight: 900;
            color: #ffffff;
            line-height: 1;
            text-shadow: 0 0 25px rgba(72, 245, 181, 0.45);
        }}

        .current-condition {{
            color: #a7f3d0;
            font-size: 20px;
            font-weight: 700;
            margin-top: 8px;
        }}

        /* 7-Day Forecast Cards */
        .forecast-card {{
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 18px;
            padding: 16px 8px;
            text-align: center;
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
            transition: all 0.25s ease;
        }}
        .forecast-card:hover {{
            transform: translateY(-4px);
            border-color: rgba(72, 245, 181, 0.5);
            background: rgba(255, 255, 255, 0.12);
        }}

        .forecast-date {{ color: #ffffff; font-size: 13px; font-weight: 700; }}
        .forecast-icon {{ font-size: 32px; margin: 10px 0; }}
        .forecast-temp {{ color: #ffffff; font-weight: 800; font-size: 18px; }}

        /* Activity Matrix Card */
        .activity-card {{
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 15px;
        }}
        .activity-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
            margin-top: 6px;
        }}
        .badge-safe {{ background: rgba(72, 245, 181, 0.2); color: #48f5b5; border: 1px solid rgba(72, 245, 181, 0.4); }}
        .badge-warning {{ background: rgba(251, 191, 36, 0.2); color: #fde047; border: 1px solid rgba(251, 191, 36, 0.4); }}
        .badge-danger {{ background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }}

        .section-heading {{
            color: #ffffff;
            font-size: 21px;
            font-weight: 800;
            margin: 28px 0 14px 0;
            display: flex;
            align-items: center;
            gap: 8px;
            text-shadow: 0 2px 8px rgba(0,0,0,0.4);
        }}

        div[data-testid="stPlotlyChart"] {{
            background: rgba(255, 255, 255, 0.08) !important;
            backdrop-filter: blur(14px) !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
            border-radius: 18px !important;
            padding: 10px !important;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.28) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_weather():
    load_glass_design()

    # 1. Hero Banner
    st.markdown(
        """
        <div class="weather-hero">
            <h1>🌿 AgriWeather Intelligence</h1>
            <p>
                Nature-inspired real-time agro-meteorological forecasting for smart farm management. 
                Live satellite conditions, 7-day crop outlook, and precision farming advisories.
            </p>
            <div class="weather-pill">
                ✦ LIVE WEATHER &nbsp; | &nbsp; SMART FARMING
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Quick City Selector (Session State)
    if "current_weather_city" not in st.session_state:
        st.session_state.current_weather_city = "Nashik"

    st.markdown("<div style='font-size:12px; font-weight:700; color:#a7f3d0; margin-bottom:6px;'>⚡ QUICK REGION SELECTOR:</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        if st.button("🍇 Nashik", use_container_width=True):
            st.session_state.current_weather_city = "Nashik"
            st.rerun()
    with c2:
        if st.button("🌾 Pune", use_container_width=True):
            st.session_state.current_weather_city = "Pune"
            st.rerun()
    with c3:
        if st.button("🍊 Nagpur", use_container_width=True):
            st.session_state.current_weather_city = "Nagpur"
            st.rerun()
    with c4:
        if st.button("🎋 Kolhapur", use_container_width=True):
            st.session_state.current_weather_city = "Kolhapur"
            st.rerun()
    with c5:
        if st.button("🍓 Satara", use_container_width=True):
            st.session_state.current_weather_city = "Satara"
            st.rerun()
    with c6:
        if st.button("🌾 Solapur", use_container_width=True):
            st.session_state.current_weather_city = "Solapur"
            st.rerun()

    # 3. Glass Search Form
    with st.form("nature_weather_search"):
        col1, col2 = st.columns([4, 1.2])
        with col1:
            city_input = st.text_input(
                "📍 Search your location",
                value=st.session_state.current_weather_city,
                placeholder="Enter city or district, e.g. Pune, Latur, Jalgaon",
            ).strip()

        with col2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            search = st.form_submit_button("Get Weather ↗", use_container_width=True)

    # जर युझरने फॉर्म सबमिट केला तर शहर अपडेट करा
    if search and city_input:
        st.session_state.current_weather_city = city_input

    target_city = st.session_state.current_weather_city

    # 4. AUTO-LOAD WEATHER (पेज उघडताच थेट डेटा लोड होईल)
    try:
        with st.spinner(f"🌱 Fetching live weather for {target_city}..."):
            location = get_coordinates(target_city)

            if location is None:
                st.error("❌ Location not found. Please verify the spelling or try a nearby city.")
                return

            weather = get_weather(location["latitude"], location["longitude"])

        current = weather["current"]
        daily = weather["daily"]

        name_parts = [location["name"], location["state"], location["country"]]
        location_name = ", ".join(part for part in name_parts if part)

        code = int(current["weather_code"])
        condition, weather_icon = WEATHER_CODES.get(code, ("Clear Weather", "🌤️"))

        # Current Weather Card
        st.markdown(
            f"""
            <div class="current-weather">
                <div style="font-size:12px; font-weight:700; color:#48f5b5; letter-spacing:1px;">
                    📍 CURRENT CONDITIONS
                </div>
                <div style="font-size:24px; font-weight:800; margin-top:4px; color:#ffffff;">
                    {location_name}
                </div>
                <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px; margin-top:16px;">
                    <div>
                        <div class="current-temperature">
                            {current['temperature_2m']}°C
                        </div>
                        <div class="current-condition">
                            {weather_icon} {condition}
                        </div>
                        <div style="color:#d1fae5; font-size:13px; margin-top:6px;">
                            Feels like <b>{current['apparent_temperature']}°C</b>
                        </div>
                    </div>
                    <div class="weather-icon" style="font-size:64px;">{weather_icon}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Activity Suitability Matrix
        st.markdown('<div class="section-heading">🚜 Farm Operations Suitability Index</div>', unsafe_allow_html=True)

        wind_spd = float(current['wind_speed_10m'])
        precip = float(current['precipitation'])
        curr_temp = float(current['temperature_2m'])

        spray_status = "Safe to Spray" if (wind_spd < 15 and precip == 0) else "Postpone (Drift Risk)"
        spray_badge = "badge-safe" if "Safe" in spray_status else "badge-danger"

        fert_status = "Optimal" if precip < 2.0 else "Delay (Washout)"
        fert_badge = "badge-safe" if "Optimal" in fert_status else "badge-warning"

        irr_status = "High Need" if curr_temp >= 32 and precip == 0 else "Normal Need"
        irr_badge = "badge-warning" if "High" in irr_status else "badge-safe"

        field_status = "Good Field Traction" if precip == 0 else "Muddy Soil Risk"
        field_badge = "badge-safe" if "Good" in field_status else "badge-danger"

        act1, act2, act3, act4 = st.columns(4)
        with act1:
            st.markdown(f"""
            <div class="activity-card">
                <div style="color:#d1fae5; font-size:12px; font-weight:700;">🧪 PESTICIDE SPRAY</div>
                <div style="color:#ffffff; font-size:15px; font-weight:700; margin-top:4px;">Wind: {wind_spd} km/h</div>
                <span class="activity-badge {spray_badge}">{spray_status}</span>
            </div>
            """, unsafe_allow_html=True)
        with act2:
            st.markdown(f"""
            <div class="activity-card">
                <div style="color:#d1fae5; font-size:12px; font-weight:700;">🌱 FERTILIZER APPLICATION</div>
                <div style="color:#ffffff; font-size:15px; font-weight:700; margin-top:4px;">Precip: {precip} mm</div>
                <span class="activity-badge {fert_badge}">{fert_status}</span>
            </div>
            """, unsafe_allow_html=True)
        with act3:
            st.markdown(f"""
            <div class="activity-card">
                <div style="color:#d1fae5; font-size:12px; font-weight:700;">💧 IRRIGATION WATER</div>
                <div style="color:#ffffff; font-size:15px; font-weight:700; margin-top:4px;">Temp: {curr_temp}°C</div>
                <span class="activity-badge {irr_badge}">{irr_status}</span>
            </div>
            """, unsafe_allow_html=True)
        with act4:
            st.markdown(f"""
            <div class="activity-card">
                <div style="color:#d1fae5; font-size:12px; font-weight:700;">🚜 HARVESTING & TILLAGE</div>
                <div style="color:#ffffff; font-size:15px; font-weight:700; margin-top:4px;">Soil Condition</div>
                <span class="activity-badge {field_badge}">{field_status}</span>
            </div>
            """, unsafe_allow_html=True)

        # 7-Day Forecast Dataframe
        forecast = pd.DataFrame({
            "Date": pd.to_datetime(daily["time"]),
            "Min Temperature (°C)": daily["temperature_2m_min"],
            "Max Temperature (°C)": daily["temperature_2m_max"],
            "Rainfall (mm)": daily["precipitation_sum"],
            "Rain Probability (%)": daily["precipitation_probability_max"],
            "Max Wind (km/h)": daily["wind_speed_10m_max"],
            "UV Index": daily.get("uv_index_max", [0]*7),
            "Water Loss ET0 (mm)": daily.get("et0_fao_evapotranspiration", [3.5]*7),
            "Weather Code": daily["weather_code"],
            "Sunrise": daily["sunrise"],
            "Sunset": daily["sunset"],
        })

        forecast["Condition"] = forecast["Weather Code"].map(
            lambda v: WEATHER_CODES.get(int(v), ("Clear", "🌤️"))[0]
        )
        forecast["Icon"] = forecast["Weather Code"].map(
            lambda v: WEATHER_CODES.get(int(v), ("Clear", "🌤️"))[1]
        )

        # 7-Day Forecast Row
        st.markdown('<div class="section-heading">📅 Next 7 Days Forecast</div>', unsafe_allow_html=True)
        forecast_cols = st.columns(7)

        for col, (_, row) in zip(forecast_cols, forecast.iterrows()):
            with col:
                day_name = row["Date"].strftime("%a")
                date_text = row["Date"].strftime("%d %b")
                rain_val = row["Rainfall (mm)"]
                rain_str = f"🌧️ {rain_val:.1f} mm" if pd.notna(rain_val) else "0.0 mm"

                st.markdown(
                    f"""
                    <div class="forecast-card">
                        <div class="forecast-date">{day_name}<br><span style="font-weight:400;color:#a7f3d0;font-size:11px;">{date_text}</span></div>
                        <div class="forecast-icon">{row['Icon']}</div>
                        <div class="forecast-temp">{row['Max Temperature (°C)']:.0f}°</div>
                        <div style="color:#a7f3d0;font-size:12px;">{row['Min Temperature (°C)']:.0f}° min</div>
                        <div style="color:#7dd3fc;font-size:11px;font-weight:700;margin-top:6px;">{rain_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Weather Analytics Charts
        st.markdown('<div class="section-heading">📊 Weather Analytics</div>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["🌡️ Temperature", "🌧️ Rainfall & Evapotranspiration", "💨 Wind Speed"])

        with tab1:
            temp_data = forecast.melt(
                id_vars=["Date"],
                value_vars=["Min Temperature (°C)", "Max Temperature (°C)"],
                var_name="Type",
                value_name="Temperature (°C)",
            )
            fig_temp = px.line(
                temp_data,
                x="Date",
                y="Temperature (°C)",
                color="Type",
                markers=True,
                title="7-Day Temperature Trend",
                color_discrete_sequence=["#38bdf8", "#fbbf24"],
            )
            fig_temp.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#d1fae5",
                margin=dict(l=15, r=15, t=50, b=15),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#a7f3d0")),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#a7f3d0"))
            )
            st.plotly_chart(fig_temp, use_container_width=True)

        with tab2:
            fig_rain = go.Figure()
            fig_rain.add_trace(go.Bar(
                x=forecast["Date"],
                y=forecast["Rainfall (mm)"],
                name="Rainfall (mm)",
                marker_color="#38bdf8"
            ))
            fig_rain.add_trace(go.Scatter(
                x=forecast["Date"],
                y=forecast["Water Loss ET0 (mm)"],
                name="Crop Water Loss ET₀ (mm)",
                mode="lines+markers",
                line=dict(color="#f59e0b", width=3, dash="dot"),
                marker=dict(size=7)
            ))
            fig_rain.update_layout(
                title="Rainfall vs Crop Evaporation",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#d1fae5",
                margin=dict(l=15, r=15, t=50, b=15),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#a7f3d0")),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#a7f3d0")),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_rain, use_container_width=True)

        with tab3:
            fig_wind = px.area(
                forecast,
                x="Date",
                y="Max Wind (km/h)",
                title="Maximum Daily Wind Speed",
            )
            fig_wind.update_traces(line_color="#48f5b5", fillcolor="rgba(72, 245, 181, 0.15)")
            fig_wind.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#d1fae5",
                margin=dict(l=15, r=15, t=50, b=15),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#a7f3d0")),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#a7f3d0"))
            )
            st.plotly_chart(fig_wind, use_container_width=True)

        # Export Forecast
        st.markdown('<div class="section-heading">📥 Export Your Forecast</div>', unsafe_allow_html=True)
        csv_data = forecast[[
            "Date", "Condition", "Min Temperature (°C)", "Max Temperature (°C)",
            "Rainfall (mm)", "Rain Probability (%)", "Max Wind (km/h)", "Sunrise", "Sunset"
        ]].copy()
        csv_data["Date"] = csv_data["Date"].dt.strftime("%Y-%m-%d")

        st.download_button(
            "⬇️ Download 7-Day Forecast CSV",
            data=csv_data.to_csv(index=False).encode("utf-8"),
            file_name=f"agriweather_forecast_{target_city}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except requests.RequestException:
        st.error("⚠️ Weather service could not be reached. Please check your internet connection.")
    except Exception as exc:
        st.error(f"⚠️ Weather data could not be processed: {exc}")