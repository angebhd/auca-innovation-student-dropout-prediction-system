"""
Analytics Module for Student Data Visualization.
Provides chart generation and statistical analysis functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from src.logger import setup_logging

logger = setup_logging(__name__)


class StudentAnalytics:
    """
    Analytics engine for student data visualization and insights.
    Generates charts using Plotly for Streamlit integration.
    """
    
    # Color schemes
    COLORS = {
        'primary': '#3b82f6',
        'secondary': '#60a5fa',
        'success': '#22c55e',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'info': '#06b6d4',
        'gradient': ['#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe']
    }
    
    RISK_COLORS = {
        'Low': '#22c55e',
        'Medium': '#f59e0b', 
        'High': '#ef4444'
    }
    
    def __init__(self):
        """Initialize analytics engine."""
        pass
    
    # ==================== OVERVIEW STATISTICS ====================
    
    def get_overview_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate overview statistics from student data.
        
        Args:
            df: Student dataframe.
            
        Returns:
            Dictionary of statistics.
        """
        stats = {
            'total_students': len(df),
            'avg_gpa': round(df['current_gpa'].mean(), 2) if 'current_gpa' in df.columns else 0,
            'avg_attendance': round(df['attendance_rate'].mean(), 2) if 'attendance_rate' in df.columns else 0,
            'total_failed_courses': int(df['failed_courses'].sum()) if 'failed_courses' in df.columns else 0,
            'students_with_failures': int((df['failed_courses'] > 0).sum()) if 'failed_courses' in df.columns else 0,
        }
        
        # Gender distribution
        if 'gender' in df.columns:
            stats['gender_distribution'] = df['gender'].value_counts().to_dict()
        
        # Scholarship stats
        if 'scholarship_status' in df.columns:
            scholarship_count = (df['scholarship_status'] == 'Yes').sum()
            stats['scholarship_students'] = int(scholarship_count)
            stats['scholarship_pct'] = round(scholarship_count / len(df) * 100, 1)
        
        return stats
    
    # ==================== CHART GENERATORS ====================
    
    def create_gpa_distribution_chart(self, df: pd.DataFrame):
        """Create GPA distribution histogram."""
        import plotly.express as px
        
        if 'current_gpa' not in df.columns:
            return None
        
        fig = px.histogram(
            df, 
            x='current_gpa',
            nbins=20,
            title='GPA Distribution',
            labels={'current_gpa': 'Current GPA', 'count': 'Number of Students'},
            color_discrete_sequence=[self.COLORS['primary']]
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font_size=18,
            showlegend=False
        )
        
        fig.add_vline(x=df['current_gpa'].mean(), line_dash="dash", 
                      line_color=self.COLORS['warning'], 
                      annotation_text=f"Avg: {df['current_gpa'].mean():.2f}")
        
        return fig
    
    def create_attendance_distribution_chart(self, df: pd.DataFrame):
        """Create attendance rate distribution."""
        import plotly.express as px
        
        if 'attendance_rate' not in df.columns:
            return None
        
        fig = px.histogram(
            df,
            x='attendance_rate',
            nbins=20,
            title='Attendance Rate Distribution',
            labels={'attendance_rate': 'Attendance Rate (%)', 'count': 'Number of Students'},
            color_discrete_sequence=[self.COLORS['info']]
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font_size=18
        )
        
        return fig
    
    def create_risk_distribution_chart(self, df: pd.DataFrame):
        """Create risk category pie chart."""
        import plotly.express as px
        
        if 'risk_category' not in df.columns:
            return None
        
        risk_counts = df['risk_category'].value_counts().reset_index()
        risk_counts.columns = ['Risk Category', 'Count']
        
        fig = px.pie(
            risk_counts,
            values='Count',
            names='Risk Category',
            title='Risk Category Distribution',
            color='Risk Category',
            color_discrete_map=self.RISK_COLORS,
            hole=0.4
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font_size=18
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def create_gpa_vs_attendance_scatter(self, df: pd.DataFrame):
        """Create GPA vs Attendance scatter plot."""
        import plotly.express as px
        
        if 'current_gpa' not in df.columns or 'attendance_rate' not in df.columns:
            return None
        
        color_col = 'risk_category' if 'risk_category' in df.columns else None
        
        fig = px.scatter(
            df,
            x='attendance_rate',
            y='current_gpa',
            color=color_col,
            color_discrete_map=self.RISK_COLORS if color_col else None,
            title='GPA vs Attendance Rate',
            labels={
                'attendance_rate': 'Attendance Rate (%)',
                'current_gpa': 'Current GPA',
                'risk_category': 'Risk Category'
            },
            opacity=0.6
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font_size=18
        )
        
        return fig
    
    def create_semester_gpa_trend(self, df: pd.DataFrame):
        """Create semester GPA trend box plot."""
        import plotly.graph_objects as go
        
        semester_cols = ['semester_1_gpa', 'semester_2_gpa', 'semester_3_gpa', 'current_gpa']
        available_cols = [col for col in semester_cols if col in df.columns]
        
        if not available_cols:
            return None
        
        fig = go.Figure()
        
        labels = ['Sem 1', 'Sem 2', 'Sem 3', 'Current']
        colors = self.COLORS['gradient']
        
        for i, col in enumerate(available_cols):
            fig.add_trace(go.Box(
                y=df[col].dropna(),
                name=labels[i] if i < len(labels) else col,
                marker_color=colors[i % len(colors)],
                boxmean=True
            ))
        
        fig.update_layout(
            title='GPA Trend Across Semesters',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font_size=18,
            yaxis_title='GPA',
            showlegend=False
        )
        
        return fig
    
    def create_failed_courses_chart(self, df: pd.DataFrame):
        """Create failed courses distribution bar chart."""
        import plotly.express as px
        
        if 'failed_courses' not in df.columns:
            return None
        
        failed_dist = df['failed_courses'].value_counts().sort_index().reset_index()
        failed_dist.columns = ['Failed Courses', 'Students']
        
        fig = px.bar(
            failed_dist,
            x='Failed Courses',
            y='Students',
            title='Failed Courses Distribution',
            color='Students',
            color_continuous_scale=['#22c55e', '#f59e0b', '#ef4444']
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font_size=18,
            showlegend=False
        )
        
        return fig
    
    def create_engagement_radar(self, df: pd.DataFrame):
        """Create engagement metrics radar chart."""
        import plotly.graph_objects as go
        
        engagement_cols = {
            'library_visits': 'Library Visits',
            'online_portal_logins': 'Portal Logins',
            'participation_score': 'Participation',
            'extracurricular_activities': 'Activities'
        }
        
        available = {k: v for k, v in engagement_cols.items() if k in df.columns}
        
        if len(available) < 3:
            return None
        
        # Normalize values (0-1 scale)
        values = []
        labels = []
        for col, label in available.items():
            col_data = df[col].dropna()
            if len(col_data) > 0 and col_data.max() > 0:
                normalized = col_data.mean() / col_data.max()
            else:
                normalized = 0
            values.append(round(normalized, 2))
            labels.append(label)
        
        # Close the radar
        values.append(values[0])
        labels.append(labels[0])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.3)',
            line=dict(color=self.COLORS['primary'], width=2),
            name='Engagement'
        ))
        
        fig.update_layout(
            title='Student Engagement Profile',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font_size=18,
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            ),
            showlegend=False
        )
        
        return fig
    
    def create_feature_importance_chart(self, importance_df: pd.DataFrame):
        """Create feature importance horizontal bar chart."""
        import plotly.express as px
        
        if importance_df is None or len(importance_df) == 0:
            return None
        
        # Take top 10
        top_features = importance_df.head(10).sort_values('importance')
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Predictive Features',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale=['#93c5fd', '#3b82f6', '#1d4ed8']
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font_size=18,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_risk_by_category_chart(self, df: pd.DataFrame, category_col: str):
        """Create risk breakdown by categorical variable."""
        import plotly.express as px
        
        if category_col not in df.columns or 'risk_category' not in df.columns:
            return None
        
        grouped = df.groupby([category_col, 'risk_category']).size().reset_index(name='count')
        
        fig = px.bar(
            grouped,
            x=category_col,
            y='count',
            color='risk_category',
            color_discrete_map=self.RISK_COLORS,
            title=f'Risk Distribution by {category_col.replace("_", " ").title()}',
            barmode='group'
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font_size=18,
            legend_title='Risk Category'
        )
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame):
        """Create correlation heatmap for numeric features."""
        import plotly.express as px
        
        numeric_cols = ['current_gpa', 'attendance_rate', 'failed_courses', 
                       'absences_count', 'late_submissions', 'participation_score']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return None
        
        corr_matrix = df[available_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title='Feature Correlation Matrix',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            text_auto='.2f'
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            title_font_size=18
        )
        
        return fig
    
    # ==================== SUMMARY GENERATORS ====================
    
    def get_top_at_risk_students(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Get top N students at highest risk."""
        if 'risk_score' not in df.columns:
            return pd.DataFrame()
        
        display_cols = ['student_id', 'current_gpa', 'attendance_rate', 
                       'failed_courses', 'risk_score', 'risk_category']
        available_cols = [col for col in display_cols if col in df.columns]
        
        return df.nlargest(n, 'risk_score')[available_cols]
    
    def get_intervention_recommendations(self, risk_summary: Dict[str, Any]) -> List[str]:
        """Generate intervention recommendations based on risk analysis."""
        recommendations = []
        
        if risk_summary.get('high_risk_pct', 0) > 20:
            recommendations.append("🚨 Critical: Over 20% of students are high-risk. Consider institution-wide support programs.")
        
        if risk_summary.get('high_risk_pct', 0) > 10:
            recommendations.append("⚠️ Prioritize one-on-one counseling for high-risk students.")
            recommendations.append("📚 Implement targeted tutoring programs for struggling students.")
        
        if risk_summary.get('avg_risk_score', 0) > 50:
            recommendations.append("📊 Average risk is elevated. Review course difficulty and support resources.")
        
        if risk_summary.get('medium_risk_pct', 0) > 30:
            recommendations.append("👀 Large medium-risk population. Proactive monitoring recommended.")
        
        if not recommendations:
            recommendations.append("✅ Risk levels are within acceptable range. Continue monitoring.")
        
        return recommendations


# Singleton instance
analytics = StudentAnalytics()
