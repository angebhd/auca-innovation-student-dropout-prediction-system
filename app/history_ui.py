"""
History UI components for activity tracking display.
"""

import streamlit as st
from datetime import datetime
from src.history import tracker


def render_history_page():
    """Render the history/activity log page."""
    st.markdown("""
        <style>
        .event-card {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        .event-type {
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            display: inline-block;
            margin-bottom: 0.5rem;
        }
        .type-data-upload { background: #3b82f6; color: white; }
        .type-data-clean { background: #22c55e; color: white; }
        .type-model-train { background: #f59e0b; color: black; }
        .type-batch-predict { background: #8b5cf6; color: white; }
        .type-single-predict { background: #06b6d4; color: white; }
        .type-user-login { background: #64748b; color: white; }
        .type-user-logout { background: #475569; color: white; }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="page-title">Activity History</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Track system activities, data processing, and model training events</p>', unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        filter_type = st.selectbox(
            "Filter by Type",
            ["All Events"] + list(tracker.EVENT_TYPES.values()),
            key="history_filter"
        )
    
    with col2:
        limit = st.slider("Show Events", 10, 100, 25)
    
    with col3:
        if st.button("Refresh"):
            st.rerun()
    
    # Get events
    event_type = None
    if filter_type != "All Events":
        # Reverse lookup type key
        for key, val in tracker.EVENT_TYPES.items():
            if val == filter_type:
                event_type = key
                break
    
    events = tracker.get_events(event_type=event_type, limit=limit)
    
    # Stats summary
    stats = tracker.get_stats()
    st.markdown("### Activity Summary")
    
    stat_cols = st.columns(4)
    stat_cols[0].metric("Total Events", stats['total_events'])
    stat_cols[1].metric("Data Operations", 
                        stats['by_type'].get('DATA_CLEAN', 0) + stats['by_type'].get('DATA_UPLOAD', 0))
    stat_cols[2].metric("Predictions", 
                        stats['by_type'].get('BATCH_PREDICT', 0) + stats['by_type'].get('SINGLE_PREDICT', 0))
    stat_cols[3].metric("Model Trainings", stats['by_type'].get('MODEL_TRAIN', 0))
    
    st.divider()
    
    # Event list
    st.markdown("### Recent Activity")
    
    if not events:
        st.info("No activity recorded yet. Start using the system to see events here.")
    else:
        for event in events:
            render_event_card(event)
    
    # Clear history (admin only)
    st.divider()
    with st.expander("Administration"):
        st.warning("This will permanently delete all activity history.")
        if st.button("Clear All History", type="secondary"):
            tracker.clear_history()
            st.success("History cleared.")
            st.rerun()


def render_event_card(event: dict):
    """Render a single event card."""
    event_type = event.get('type', 'UNKNOWN')
    type_label = event.get('type_label', event_type)
    description = event.get('description', 'No description')
    timestamp = event.get('timestamp', '')
    user = event.get('user', 'System')
    
    # Format timestamp
    try:
        dt = datetime.fromisoformat(timestamp)
        time_str = dt.strftime("%b %d, %Y %H:%M")
    except:
        time_str = timestamp
    
    # Type badge styling
    type_styles = {
        'DATA_UPLOAD': '#3b82f6',
        'DATA_CLEAN': '#22c55e',
        'MODEL_TRAIN': '#f59e0b',
        'BATCH_PREDICT': '#8b5cf6',
        'SINGLE_PREDICT': '#06b6d4',
        'USER_LOGIN': '#64748b',
        'USER_LOGOUT': '#475569',
        'EXPORT_DATA': '#f97316'
    }
    
    badge_color = type_styles.get(event_type, '#64748b')
    
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
                <span style="background: {badge_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">{type_label}</span>  
                <br><span style="color: #1e293b; font-weight: 500;">{description}</span>  
                <br><small style="color: #64748b;">{user or 'System'} | {time_str}</small>
            """, unsafe_allow_html=True)
        with col2:
            # Show details if available
            details = event.get('details', {})
            if details:
                with st.popover("Details"):
                    st.json(details)
        st.markdown("---")


def render_activity_widget():
    """Render a compact activity widget for the sidebar or dashboard."""
    events = tracker.get_recent_activity(limit=5)
    
    st.markdown("#### Recent Activity")
    
    if not events:
        st.caption("No recent activity")
    else:
        for event in events:
            try:
                dt = datetime.fromisoformat(event.get('timestamp', ''))
                time_str = dt.strftime("%H:%M")
            except:
                time_str = "..."
            
            st.caption(f"{event.get('description', 'Event')[:40]}... ({time_str})")
