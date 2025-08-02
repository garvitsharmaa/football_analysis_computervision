import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans

# --------------------------------------------------
# Data Loading and Caching
# --------------------------------------------------
@st.cache_data
def load_and_clean_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    # Boolean conversion for possession
    df['Has Ball'] = df.get('Has Ball', False).fillna(False).astype(bool)

    # Handle speed - convert from m/s to km/h if needed
    if 'Speed (km/h)' in df.columns:
        df['Speed_kmh'] = pd.to_numeric(df['Speed (km/h)'], errors='coerce')
    elif 'Speed (m/s)' in df.columns:
        # Convert from m/s to km/h (1 m/s = 3.6 km/h)
        df['Speed_kmh'] = pd.to_numeric(df['Speed (m/s)'], errors='coerce') * 3.6
    else:
        raise KeyError("Could not find 'Speed (km/h)' or 'Speed (m/s)' column in data.")

    # Handle distance - ensure it's in meters
    if 'Distance (m)' in df.columns:
        df['Distance'] = pd.to_numeric(df['Distance (m)'], errors='coerce')
    elif 'Distance (km)' in df.columns:
        # Convert from km to m if necessary
        df['Distance'] = pd.to_numeric(df['Distance (km)'], errors='coerce') * 1000
    elif 'Distance' in df.columns:
        # Assume it's already in meters
        df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
    else:
        raise KeyError("Could not find distance column in data.")

    # Only drop rows where X and Y are missing (keep rows with missing speed/distance for some visualizations)
    df_with_coords = df.dropna(subset=['X', 'Y'])
    
    # Add Player Name column - ensuring it exists in the returned dataframe
    player_ids = df_with_coords['Object ID'].unique()
    df_with_coords['Player Name'] = df_with_coords['Object ID'].apply(lambda x: f"Player #{x}")
    
    return df_with_coords

# Sidebar: File uploader and filters
st.sidebar.title("Data & Filters")
file = st.sidebar.file_uploader("Upload tracking_data.csv", type=['csv'])
if not file:
    st.sidebar.info("Please upload a tracking_data.csv file to proceed.")
    st.stop()

# Load and preprocess data
df = load_and_clean_data(file)

# Frame range slider
total_min, total_max = int(df['Frame'].min()), int(df['Frame'].max())
start_frame, end_frame = st.sidebar.slider(
    "Frame Range", total_min, total_max, (total_min, total_max)
)
df = df[df['Frame'].between(start_frame, end_frame)]

# Speed threshold slider (km/h)
max_speed_kmh = st.sidebar.slider(
    "Max Realistic Speed (km/h)", 10.0, 100.0, 60.0
)

# Separate players and ball data
# Make sure we only consider entries with 'Player' in Object Type for players
players = df[df['Object Type'] == 'Player'].copy()
# Apply speed threshold only to players
players = players[(players['Speed_kmh'] <= max_speed_kmh) | players['Speed_kmh'].isna()]
# Ball data would be identified separately if present
ball = df[df['Object Type'] == 'Ball'].copy() if 'Ball' in df['Object Type'].unique() else pd.DataFrame()

# Team selection
teams = ['All'] + sorted(players['Team'].dropna().unique().astype(str).tolist())
selected_team = st.sidebar.selectbox("Team", teams)
if selected_team != 'All':
    players = players[players['Team'].astype(str) == selected_team]

# Make sure 'Player Name' exists before trying to use it
if 'Player Name' not in players.columns:
    players['Player Name'] = players['Object ID'].apply(lambda x: f"Player #{x}")

# Player selection
names = ['All'] + sorted(players['Player Name'].unique().tolist())
selected_name = st.sidebar.selectbox("Player", names)
if selected_name != 'All':
    players = players[players['Player Name'] == selected_name]

# App title and description
st.title("⚽ Advanced Football Match Dashboard")
st.markdown("Built by Garvit Sharma — unit detection & robust handling.")

# Define tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Match Overview", "Possession", "Heatmap", "Player Details", "Advanced Metrics"
])

# Tab 1: Match Overview
with tab1:
    st.subheader("Match Summary")
    total_players = players['Object ID'].nunique()
    
    # Calculate total distance in meters (sum only valid distance values)
    valid_distances = players['Distance'].dropna()
    total_distance = valid_distances.sum() if not valid_distances.empty else 0
    
    # Calculate maximum speed per player, then find overall maximum
    player_max_speeds = players.groupby('Object ID')['Speed_kmh'].max().dropna()
    top_speed = player_max_speeds.max() if not player_max_speeds.empty else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Players", total_players)
    c2.metric("Total Distance (m)", f"{total_distance:.1f}")
    c3.metric("Top Speed (km/h)", f"{top_speed:.1f}")

    # Top 5 Sprint Speeds - only consider players with valid speed data
    players_with_speed = players.dropna(subset=['Speed_kmh'])
    if not players_with_speed.empty:
        top_fast = players_with_speed.groupby('Player Name')['Speed_kmh'].max().nlargest(5).reset_index()
        fig_fast = px.bar(
            top_fast, x='Speed_kmh', y='Player Name', orientation='h',
            title='Top 5 Sprint Speeds (km/h)', color='Speed_kmh',
            color_continuous_scale='Turbo'
        )
        st.plotly_chart(fig_fast, use_container_width=True)
    else:
        st.info("No valid speed data available for players.")

# Tab 2: Possession
with tab2:
    st.subheader("Ball Possession Share")
    possession_data = players[players['Has Ball']].copy()
    if not possession_data.empty:
        possession = possession_data.groupby('Team').size()
        fig_pos = px.pie(
            names=possession.index, values=possession.values,
            title='Possession by Team (Events Count)'
        )
        st.plotly_chart(fig_pos)
    else:
        st.info("No ball possession events in the selected range.")

# Tab 3: Heatmap of Movements
with tab3:
    st.subheader("Player Movement Heatmap")
    if not players.empty:
        heat = px.density_heatmap(
            players, x='X', y='Y', nbinsx=60, nbinsy=40,
            title='Heatmap of Player Positions', color_continuous_scale='Viridis'
        )
        heat.update_layout(yaxis_autorange='reversed')
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("No player position data available.")

# Tab 4: Individual Player Details
with tab4:
    if selected_name == 'All':
        st.info("Select a player to view detailed stats.")
    else:
        player_data = players[players['Player Name'] == selected_name]
        
        # Calculate total distance for selected player (valid values only)
        player_dist = player_data['Distance'].dropna().sum()
        
        # Get max speed for selected player
        player_speed = player_data['Speed_kmh'].dropna().max() if not player_data['Speed_kmh'].dropna().empty else 0
        
        col1, col2 = st.columns(2)
        col1.metric("Distance Covered (m)", f"{player_dist:.1f}")
        col2.metric("Max Speed (km/h)", f"{player_speed:.1f}")

        # Only show speed timeline if we have valid data
        if not player_data['Speed_kmh'].dropna().empty:
            st.subheader("Speed Over Time")
            fig_line = px.line(
                player_data.sort_values('Frame'), x='Frame', y='Speed_kmh',
                title=f"{selected_name} — Speed Timeline (km/h)"
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No speed data available for this player.")

# Tab 5: Advanced Metrics
with tab5:
    st.subheader("Sprints & Positional Clustering")
    
    # Only calculate sprints if we have valid speed data
    valid_speeds = players['Speed_kmh'].dropna()
    if not valid_speeds.empty:
        sprint_threshold = valid_speeds.quantile(0.75)
        sprint_count = (valid_speeds > sprint_threshold).sum()
        st.metric("Sprint Count (>75th pct)", sprint_count)
    else:
        st.info("No valid speed data available for sprint analysis.")

    # Only do clustering if we have enough valid position data
    valid_coords = players.dropna(subset=['X', 'Y'])
    if len(valid_coords) >= 3:
        coords = valid_coords[['X', 'Y']]
        kmeans = KMeans(n_clusters=min(3, len(coords)), random_state=42).fit(coords)
        valid_coords['Cluster'] = kmeans.labels_
        fig_clust = px.scatter(
            valid_coords, x='X', y='Y', color='Cluster',
            title='Positional Clusters on Pitch'
        )
        st.plotly_chart(fig_clust, use_container_width=True)
    else:
        st.info("Not enough valid position data points for clustering.")

# Footer
st.markdown("---")
st.write("© 2025 Garvit Sharma")