"""
Professional Dashboard Component for Student Dropout Prediction System.
Provides executive summary, KPIs, and actionable insights.
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any, Optional


class DashboardService:
    """Service class for dashboard data and insights generation."""
    
    def __init__(self, cleaned_data_path: str = "data/cleaned"):
        self.cleaned_data_path = cleaned_data_path
    
    def get_latest_dataset(self) -> Optional[pd.DataFrame]:
        """Get the most recently modified cleaned dataset."""
        if not os.path.exists(self.cleaned_data_path):
            return None
        
        files = [f for f in os.listdir(self.cleaned_data_path) if f.endswith('.csv')]
        if not files:
            return None
        
        # Get most recent file
        latest = max(files, key=lambda x: os.path.getmtime(os.path.join(self.cleaned_data_path, x)))
        return pd.read_csv(os.path.join(self.cleaned_data_path, latest))
    
    def calculate_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key performance indicators."""
        kpis = {
            'total_students': len(df),
            'avg_gpa': round(df['current_gpa'].mean(), 2) if 'current_gpa' in df.columns else 0,
            'avg_attendance': round(df['attendance_rate'].mean(), 1) if 'attendance_rate' in df.columns else 0,
            'at_risk_count': 0,
            'at_risk_pct': 0,
            'intervention_needed': 0
        }
        
        if 'risk_category' in df.columns:
            high_risk = (df['risk_category'] == 'High').sum()
            medium_risk = (df['risk_category'] == 'Medium').sum()
            kpis['at_risk_count'] = high_risk
            kpis['at_risk_pct'] = round(high_risk / len(df) * 100, 1)
            kpis['intervention_needed'] = high_risk + medium_risk
        
        return kpis
    
    def generate_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> list:
        """Generate actionable insights based on data analysis."""
        insights = []
        
        # GPA Analysis
        if 'current_gpa' in df.columns:
            low_gpa_count = (df['current_gpa'] < 10).sum()
            low_gpa_pct = round(low_gpa_count / len(df) * 100, 1)
            if low_gpa_pct > 15:
                insights.append({
                    'type': 'warning',
                    'title': 'Academic Performance Concern',
                    'message': f'{low_gpa_pct}% of students have GPA below 10.0. Consider implementing academic support programs.',
                    'metric': f'{low_gpa_count} students',
                    'priority': 'High'
                })
            elif low_gpa_pct > 5:
                insights.append({
                    'type': 'info',
                    'title': 'Academic Performance Note',
                    'message': f'{low_gpa_pct}% of students have GPA below 10.0. Monitor these students for potential intervention.',
                    'metric': f'{low_gpa_count} students',
                    'priority': 'Medium'
                })
        
        # Attendance Analysis
        if 'attendance_rate' in df.columns:
            low_attendance = (df['attendance_rate'] < 70).sum()
            low_att_pct = round(low_attendance / len(df) * 100, 1)
            if low_att_pct > 20:
                insights.append({
                    'type': 'warning',
                    'title': 'Attendance Alert',
                    'message': f'{low_att_pct}% of students have attendance below 70%. Investigate attendance barriers.',
                    'metric': f'{low_attendance} students',
                    'priority': 'High'
                })
        
        # Failed Courses
        if 'failed_courses' in df.columns:
            students_with_failures = (df['failed_courses'] > 0).sum()
            multiple_failures = (df['failed_courses'] >= 2).sum()
            if multiple_failures > 0:
                insights.append({
                    'type': 'warning',
                    'title': 'Course Failure Pattern',
                    'message': f'{multiple_failures} students have failed 2 or more courses. Prioritize these for academic counseling.',
                    'metric': f'{multiple_failures} students',
                    'priority': 'High'
                })
        
        # Risk Distribution
        if 'risk_category' in df.columns:
            high_risk_pct = kpis.get('at_risk_pct', 0)
            if high_risk_pct > 15:
                insights.append({
                    'type': 'critical',
                    'title': 'Elevated Dropout Risk',
                    'message': f'{high_risk_pct}% of students are classified as high-risk. Immediate intervention program recommended.',
                    'metric': f'{kpis.get("at_risk_count", 0)} students',
                    'priority': 'Critical'
                })
        
        # Positive insights
        if kpis.get('avg_gpa', 0) >= 14:
            insights.append({
                'type': 'success',
                'title': 'Strong Academic Performance',
                'message': f'Average GPA of {kpis["avg_gpa"]} indicates healthy overall academic performance.',
                'metric': f'{kpis["avg_gpa"]} avg GPA',
                'priority': 'Info'
            })
        
        if kpis.get('avg_attendance', 0) >= 85:
            insights.append({
                'type': 'success',
                'title': 'Good Attendance Rates',
                'message': f'Average attendance of {kpis["avg_attendance"]}% shows strong student engagement.',
                'metric': f'{kpis["avg_attendance"]}% avg',
                'priority': 'Info'
            })
        
        return insights
    
    def get_recommendations(self, kpis: Dict[str, Any]) -> list:
        """Generate strategic recommendations based on KPIs."""
        recommendations = []
        
        if kpis.get('at_risk_pct', 0) > 10:
            recommendations.append({
                'action': 'Implement Early Warning System',
                'description': 'Set up automated alerts for students showing declining performance patterns.',
                'impact': 'High',
                'timeline': 'Immediate'
            })
        
        if kpis.get('avg_attendance', 100) < 80:
            recommendations.append({
                'action': 'Attendance Improvement Initiative',
                'description': 'Investigate root causes of absenteeism and implement targeted interventions.',
                'impact': 'Medium',
                'timeline': '2-4 weeks'
            })
        
        if kpis.get('intervention_needed', 0) > 0:
            recommendations.append({
                'action': 'Schedule Counseling Sessions',
                'description': f'Prioritize {kpis["intervention_needed"]} students for one-on-one academic counseling.',
                'impact': 'High',
                'timeline': 'This week'
            })
        
        recommendations.append({
            'action': 'Regular Model Retraining',
            'description': 'Update prediction model monthly with new student data for improved accuracy.',
            'impact': 'Medium',
            'timeline': 'Monthly'
        })
        
        return recommendations


def render_professional_dashboard():
    """Render the professional executive dashboard."""
    from src.prediction import predictor
    from src.analytics import analytics
    
    # Header
    st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h1 style="font-size: 2.2rem; font-weight: 700; color: #1e293b; margin-bottom: 0.25rem;">
                Student Success Dashboard
            </h1>
            <p style="color: #64748b; font-size: 1rem; margin: 0;">
                Executive overview of student performance and risk indicators
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize dashboard service
    dashboard = DashboardService()
    df = dashboard.get_latest_dataset()
    
    if df is None:
        st.warning("No data available. Please upload and process student data in Data Management.")
        _render_getting_started()
        return
    
    # Run predictions if model is trained and predictions don't exist
    if 'risk_category' not in df.columns and predictor.is_trained:
        df = predictor.predict(df)
    
    # Calculate KPIs
    kpis = dashboard.calculate_kpis(df)
    
    # KPI Cards Row
    st.markdown("### Key Performance Indicators")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        _render_kpi_card(
            "Total Students",
            str(kpis['total_students']),
            "Active in dataset",
            "#3b82f6"
        )
    
    with kpi_col2:
        _render_kpi_card(
            "Average GPA",
            f"{kpis['avg_gpa']}/20",
            "Current semester",
            "#22c55e" if kpis['avg_gpa'] >= 12 else "#f59e0b"
        )
    
    with kpi_col3:
        _render_kpi_card(
            "Attendance Rate",
            f"{kpis['avg_attendance']}%",
            "Class average",
            "#22c55e" if kpis['avg_attendance'] >= 80 else "#f59e0b"
        )
    
    with kpi_col4:
        _render_kpi_card(
            "At-Risk Students",
            str(kpis['at_risk_count']),
            f"{kpis['at_risk_pct']}% of total",
            "#ef4444" if kpis['at_risk_pct'] > 10 else "#f59e0b"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two Column Layout - Charts and Insights
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("### Risk Distribution")
        if 'risk_category' in df.columns:
            _render_risk_summary_chart(df)
        else:
            st.info("Train the prediction model in Risk Assessment to view risk distribution.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Performance Trends")
        _render_performance_overview(df)
    
    with col_right:
        # Insights Panel
        st.markdown("### Actionable Insights")
        insights = dashboard.generate_insights(df, kpis)
        
        if insights:
            for insight in insights[:4]:  # Show top 4 insights
                _render_insight_card(insight)
        else:
            st.info("No significant insights detected. Data appears within normal parameters.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("### Recommendations")
        recommendations = dashboard.get_recommendations(kpis)
        for rec in recommendations[:3]:  # Show top 3
            _render_recommendation_card(rec)
    
    # Students Requiring Attention
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Students Requiring Immediate Attention")
    
    if 'risk_score' in df.columns:
        high_risk_students = df[df['risk_category'] == 'High'].nlargest(5, 'risk_score')
        if not high_risk_students.empty:
            display_cols = ['student_id', 'current_gpa', 'attendance_rate', 'failed_courses', 'risk_score']
            available_cols = [c for c in display_cols if c in high_risk_students.columns]
            
            # Style the dataframe
            st.dataframe(
                high_risk_students[available_cols].style.background_gradient(
                    subset=['risk_score'] if 'risk_score' in available_cols else [],
                    cmap='Reds'
                ),
                use_container_width=True,
                hide_index=True
            )
            
            st.caption("Students sorted by risk score. Higher scores indicate greater dropout probability.")
        else:
            st.success("No students currently classified as high-risk.")
    else:
        st.info("Run risk assessment to identify students requiring attention.")


def _render_kpi_card(title: str, value: str, subtitle: str, color: str):
    """Render a KPI metric card."""
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}15, {color}05);
            border-left: 4px solid {color};
            padding: 1.25rem;
            border-radius: 8px;
            height: 100%;
        ">
            <p style="color: #64748b; font-size: 0.85rem; margin: 0 0 0.5rem 0; text-transform: uppercase; letter-spacing: 0.5px;">
                {title}
            </p>
            <p style="color: #1e293b; font-size: 1.75rem; font-weight: 700; margin: 0 0 0.25rem 0;">
                {value}
            </p>
            <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">
                {subtitle}
            </p>
        </div>
    """, unsafe_allow_html=True)


def _render_insight_card(insight: dict):
    """Render an insight card."""
    colors = {
        'critical': ('#ef4444', '#fef2f2'),
        'warning': ('#f59e0b', '#fffbeb'),
        'info': ('#3b82f6', '#eff6ff'),
        'success': ('#22c55e', '#f0fdf4')
    }
    
    border_color, bg_color = colors.get(insight['type'], ('#64748b', '#f8fafc'))
    
    st.markdown(f"""
        <div style="
            background: {bg_color};
            border-left: 3px solid {border_color};
            padding: 0.875rem;
            border-radius: 6px;
            margin-bottom: 0.75rem;
        ">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <p style="color: #1e293b; font-weight: 600; font-size: 0.9rem; margin: 0 0 0.25rem 0;">
                    {insight['title']}
                </p>
                <span style="
                    background: {border_color};
                    color: white;
                    font-size: 0.65rem;
                    padding: 0.125rem 0.5rem;
                    border-radius: 4px;
                    font-weight: 600;
                ">{insight['priority']}</span>
            </div>
            <p style="color: #475569; font-size: 0.8rem; margin: 0; line-height: 1.4;">
                {insight['message']}
            </p>
        </div>
    """, unsafe_allow_html=True)


def _render_recommendation_card(rec: dict):
    """Render a recommendation card."""
    impact_colors = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#22c55e'}
    
    st.markdown(f"""
        <div style="
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 0.875rem;
            border-radius: 6px;
            margin-bottom: 0.75rem;
        ">
            <p style="color: #1e293b; font-weight: 600; font-size: 0.85rem; margin: 0 0 0.25rem 0;">
                {rec['action']}
            </p>
            <p style="color: #64748b; font-size: 0.75rem; margin: 0 0 0.5rem 0; line-height: 1.4;">
                {rec['description']}
            </p>
            <div style="display: flex; gap: 1rem; font-size: 0.7rem;">
                <span style="color: {impact_colors.get(rec['impact'], '#64748b')};">
                    Impact: {rec['impact']}
                </span>
                <span style="color: #64748b;">
                    Timeline: {rec['timeline']}
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def _render_risk_summary_chart(df: pd.DataFrame):
    """Render a summary risk chart."""
    import plotly.graph_objects as go
    
    risk_counts = df['risk_category'].value_counts()
    
    colors = {'Low': '#22c55e', 'Medium': '#f59e0b', 'High': '#ef4444'}
    
    fig = go.Figure(data=[go.Bar(
        x=list(risk_counts.index),
        y=list(risk_counts.values),
        marker_color=[colors.get(cat, '#64748b') for cat in risk_counts.index],
        text=list(risk_counts.values),
        textposition='auto',
    )])
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Risk Category', gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(title='Number of Students', gridcolor='rgba(0,0,0,0.05)'),
        font=dict(family="Inter, sans-serif", color='#475569')
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_performance_overview(df: pd.DataFrame):
    """Render performance overview metrics."""
    import plotly.graph_objects as go
    
    if 'current_gpa' not in df.columns:
        return
    
    # GPA distribution bins
    bins = [0, 8, 10, 12, 14, 16, 20]
    labels = ['0-8', '8-10', '10-12', '12-14', '14-16', '16-20']
    df['gpa_range'] = pd.cut(df['current_gpa'], bins=bins, labels=labels, include_lowest=True)
    gpa_dist = df['gpa_range'].value_counts().sort_index()
    
    fig = go.Figure(data=[go.Bar(
        x=list(gpa_dist.index),
        y=list(gpa_dist.values),
        marker_color='#3b82f6',
        text=list(gpa_dist.values),
        textposition='auto',
    )])
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='GPA Range', gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(title='Students', gridcolor='rgba(0,0,0,0.05)'),
        font=dict(family="Inter, sans-serif", color='#475569')
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_getting_started():
    """Render getting started guide when no data is available."""
    st.markdown("""
        <div style="
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin-top: 2rem;
        ">
            <h3 style="color: #1e293b; margin-bottom: 1rem;">Getting Started</h3>
            <p style="color: #64748b; margin-bottom: 1.5rem;">
                Follow these steps to begin analyzing student dropout risk:
            </p>
            <div style="text-align: left; max-width: 400px; margin: 0 auto;">
                <p style="color: #475569; margin: 0.5rem 0;"><strong>1.</strong> Navigate to Data Management</p>
                <p style="color: #475569; margin: 0.5rem 0;"><strong>2.</strong> Upload or generate student data</p>
                <p style="color: #475569; margin: 0.5rem 0;"><strong>3.</strong> Clean and process the dataset</p>
                <p style="color: #475569; margin: 0.5rem 0;"><strong>4.</strong> Go to Risk Assessment to train the model</p>
                <p style="color: #475569; margin: 0.5rem 0;"><strong>5.</strong> Return here to view insights</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
