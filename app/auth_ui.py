"""
Authentication UI components for Streamlit.
Contains login, signup, and profile management interfaces.
"""

import streamlit as st
from src.auth import auth_service


def render_auth_styles():
    """Render custom CSS styles for authentication pages."""
    st.markdown("""
        <style>
        .auth-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem;
            border-radius: 12px;
            background-color: #1e293b;
            border: 1px solid #334155;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            margin-top: 3rem;
        }
        .auth-header {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #3b82f6, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .auth-subheader {
            font-size: 1rem;
            color: #94a3b8;
            text-align: center;
            margin-bottom: 2rem;
        }
        .auth-link {
            color: #3b82f6;
            cursor: pointer;
            text-decoration: underline;
        }
        .auth-footer {
            margin-top: 1.5rem;
            text-align: center;
            color: #94a3b8;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)


def render_login_page():
    """Render the login page with authentication."""
    render_auth_styles()
    
    _, col2, _ = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.markdown('<div class="auth-header">Umoja Platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subheader">Sign in to your account</div>', unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username or Email", placeholder="Enter your username or email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if submitted:
                if username and password:
                    success, message, user_data = auth_service.login(username, password)
                    if success:
                        st.session_state['logged_in'] = True
                        st.session_state['user'] = user_data
                        
                        # Log login event
                        from src.history import log_user_login
                        log_user_login(user_data.get('username', username))
                        
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter username and password")
        
        # Show default credentials hint
        st.markdown("---")
        st.caption("**Default credentials:** admin / admin123")
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_auth_page():
    """
    Main authentication page - shows login only.
    """
    render_login_page()


def render_logout_button():
    """Render logout button in sidebar."""
    if st.sidebar.button("Logout", key="logout_btn", type="secondary", use_container_width=True):
        # Log logout event
        if 'user' in st.session_state and st.session_state['user']:
            from src.history import log_user_logout
            log_user_logout(st.session_state['user'].get('username', 'unknown'))
        
        # Clear session state
        st.session_state['logged_in'] = False
        st.session_state['user'] = None
        st.session_state['auth_page'] = 'login'
        st.rerun()


def render_user_profile_sidebar():
    """Render user info in the sidebar."""
    if 'user' in st.session_state and st.session_state['user']:
        user = st.session_state['user']
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**👤 {user.get('full_name', user.get('username', 'User'))}**")
        st.sidebar.caption(f"@{user.get('username', 'unknown')}")
        st.sidebar.markdown("---")


def render_profile_page():
    """Render user profile management page."""
    st.header("User Profile")
    
    if 'user' not in st.session_state or not st.session_state['user']:
        st.warning("Please log in to view your profile")
        return
    
    user = st.session_state['user']
    
    # Profile Information
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Profile Picture")
        st.markdown("""
            <div style="
                width: 150px; 
                height: 150px; 
                background: linear-gradient(135deg, #3b82f6, #60a5fa);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 4rem;
                color: white;
                margin: 1rem 0;
            ">👤</div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Account Information")
        st.markdown(f"**Full Name:** {user.get('full_name', 'Not set')}")
        st.markdown(f"**Username:** @{user.get('username', 'unknown')}")
        st.markdown(f"**Email:** {user.get('email', 'Not set')}")
        st.markdown(f"**Role:** {user.get('role', 'user').capitalize()}")
        st.markdown(f"**Member Since:** {user.get('created_at', 'Unknown')[:10] if user.get('created_at') else 'Unknown'}")
    
    st.divider()
    
    # Change Password Section
    st.markdown("### Change Password")
    
    with st.form("change_password_form"):
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_new_password = st.text_input("Confirm New Password", type="password")
        
        if st.form_submit_button("Update Password"):
            if not all([current_password, new_password, confirm_new_password]):
                st.warning("Please fill in all password fields")
            elif new_password != confirm_new_password:
                st.error("New passwords do not match")
            else:
                success, message = auth_service.change_password(
                    username=user.get('username'),
                    current_password=current_password,
                    new_password=new_password
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)
