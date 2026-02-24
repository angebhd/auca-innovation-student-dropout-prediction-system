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
        /* Hide default Streamlit elements for cleaner login */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Center the login form vertically */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 0;
            max-width: 100%;
        }
        
        .login-wrapper {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 80vh;
        }
        
        .login-card {
            background: linear-gradient(145deg, #1e293b, #0f172a);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 2.5rem;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }
        
        .login-logo {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        .login-logo h1 {
            font-size: 1.8rem;
            font-weight: 700;
            color: #3b82f6;
            margin: 0;
            letter-spacing: -0.5px;
        }
        
        .login-logo p {
            color: #64748b;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        
        .credentials-hint {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 8px;
            padding: 0.75rem;
            text-align: center;
            margin-top: 1rem;
        }
        
        .credentials-hint code {
            background: rgba(59, 130, 246, 0.2);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-family: monospace;
        }
        </style>
    """, unsafe_allow_html=True)


def render_login_page():
    """Render a compact, professional login page."""
    render_auth_styles()
    
    # Create centered layout
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        # Logo and branding
        st.markdown("""
            <div class="login-logo">
                <h1>UMOJA</h1>
                <p>Student Success Prediction Platform</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
            
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if submitted:
                if username and password:
                    success, message, user_data = auth_service.login(username, password)
                    if success:
                        st.session_state['logged_in'] = True
                        st.session_state['user'] = user_data
                        
                        from src.history import log_user_login
                        log_user_login(user_data.get('username', username))
                        
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter credentials")
        
        # Credentials hint
        st.markdown("""
            <div class="credentials-hint">
                <small style="color: #64748b;">Default: <code>admin</code> / <code>admin123</code></small>
            </div>
        """, unsafe_allow_html=True)


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
        st.sidebar.markdown(f"**{user.get('full_name', user.get('username', 'User'))}**")
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
        initials = ''.join([n[0].upper() for n in user.get('full_name', 'U').split()[:2]]) or 'U'
        st.markdown(f"""
            <div style="
                width: 150px; 
                height: 150px; 
                background: linear-gradient(135deg, #3b82f6, #60a5fa);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 3rem;
                font-weight: 700;
                color: white;
                margin: 1rem 0;
            ">{initials}</div>
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
