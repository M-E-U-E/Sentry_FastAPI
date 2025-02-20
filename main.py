@app.get("/sentry-debug", tags=["Sentry"])
async def trigger_error():
    try:
        division_by_zero = 1 / 0
    except ZeroDivisionError as e:
        sentry_sdk.capture_exception(e)
        return {"error": "Division by zero", "message": str(e)}