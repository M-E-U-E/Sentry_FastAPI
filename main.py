@app.get("/sentry-debug", tags=["Sentry"])
async def trigger_error():
    try:
        division_by_zero = 1 / 0
    except ZeroDivisionError as e:
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail="Division by zero error")