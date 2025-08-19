@echo off
echo Streamlit 앱을 시작합니다...
echo 포트 8501에서 실행됩니다.
echo 브라우저에서 http://localhost:8501 을 열어주세요.
echo.
echo 종료하려면 Ctrl+C를 누르세요.
echo.
streamlit run streamlit_app.py --server.port 8501
pause

