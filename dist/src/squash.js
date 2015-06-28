// squashing functions
function LOGISTIC(x, derivate) {
    if (derivate) {
        var fx = LOGISTIC(x);
        return fx * (1 - fx);
    }
    return 1 / (1 + Math.exp(-x));
}
exports.LOGISTIC = LOGISTIC;
function TANH(x, derivate) {
    if (derivate)
        return 1 - Math.pow(TANH(x), 2);
    var eP = Math.exp(x);
    var eN = 1 / eP;
    return (eP - eN) / (eP + eN);
}
exports.TANH = TANH;
function IDENTITY(x, derivate) {
    return derivate ? 1 : x;
}
exports.IDENTITY = IDENTITY;
function HLIM(x, derivate) {
    return derivate ? 1 : +(x > 0);
}
exports.HLIM = HLIM;
function SOFTPLUS(x, derivate) {
    if (derivate)
        return 1 - 1 / (1 + Math.exp(x));
    return Math.log(1 + Math.exp(x));
}
exports.SOFTPLUS = SOFTPLUS;
function EXP(x, derivate) {
    return Math.exp(x);
}
exports.EXP = EXP;

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9zcXVhc2gudHMiXSwibmFtZXMiOlsiTE9HSVNUSUMiLCJUQU5IIiwiSURFTlRJVFkiLCJITElNIiwiU09GVFBMVVMiLCJFWFAiXSwibWFwcGluZ3MiOiJBQUVBLEFBRUEsc0JBRnNCO1NBRU4sUUFBUSxDQUFDLENBQVMsRUFBRSxRQUFrQjtJQUNyREEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsUUFBUUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7UUFDZEEsSUFBSUEsRUFBRUEsR0FBR0EsUUFBUUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7UUFDckJBLE1BQU1BLENBQUNBLEVBQUVBLEdBQUdBLENBQUNBLENBQUNBLEdBQUdBLEVBQUVBLENBQUNBLENBQUNBO0lBQ3RCQSxDQUFDQTtJQUNEQSxNQUFNQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQSxHQUFHQSxJQUFJQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtBQUUvQkEsQ0FBQ0E7QUFQZSxnQkFBUSxHQUFSLFFBT2YsQ0FBQTtBQUVELFNBQWdCLElBQUksQ0FBQyxDQUFTLEVBQUUsUUFBa0I7SUFDakRDLEVBQUVBLENBQUNBLENBQUNBLFFBQVFBLENBQUNBO1FBQ1pBLE1BQU1BLENBQUNBLENBQUNBLEdBQUdBLElBQUlBLENBQUNBLEdBQUdBLENBQUNBLElBQUlBLENBQUNBLENBQUNBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBLENBQUNBO0lBQ2pDQSxJQUFJQSxFQUFFQSxHQUFHQSxJQUFJQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtJQUNyQkEsSUFBSUEsRUFBRUEsR0FBR0EsQ0FBQ0EsR0FBR0EsRUFBRUEsQ0FBQ0E7SUFDaEJBLE1BQU1BLENBQUNBLENBQUNBLEVBQUVBLEdBQUdBLEVBQUVBLENBQUNBLEdBQUdBLENBQUNBLEVBQUVBLEdBQUdBLEVBQUVBLENBQUNBLENBQUNBO0FBQzlCQSxDQUFDQTtBQU5lLFlBQUksR0FBSixJQU1mLENBQUE7QUFFRCxTQUFnQixRQUFRLENBQUMsQ0FBUyxFQUFFLFFBQWtCO0lBQ3JEQyxNQUFNQSxDQUFDQSxRQUFRQSxHQUFHQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQTtBQUN6QkEsQ0FBQ0E7QUFGZSxnQkFBUSxHQUFSLFFBRWYsQ0FBQTtBQUVELFNBQWdCLElBQUksQ0FBQyxDQUFTLEVBQUUsUUFBa0I7SUFDakRDLE1BQU1BLENBQUNBLFFBQVFBLEdBQUdBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBLENBQUNBO0FBQ2hDQSxDQUFDQTtBQUZlLFlBQUksR0FBSixJQUVmLENBQUE7QUFFRCxTQUFnQixRQUFRLENBQUMsQ0FBUyxFQUFFLFFBQWtCO0lBQ3JEQyxFQUFFQSxDQUFDQSxDQUFDQSxRQUFRQSxDQUFDQTtRQUNaQSxNQUFNQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQSxHQUFHQSxJQUFJQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtJQUNsQ0EsTUFBTUEsQ0FBQ0EsSUFBSUEsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsSUFBSUEsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7QUFDbENBLENBQUNBO0FBSmUsZ0JBQVEsR0FBUixRQUlmLENBQUE7QUFFRCxTQUFnQixHQUFHLENBQUMsQ0FBUyxFQUFFLFFBQWtCO0lBQ2hEQyxNQUFNQSxDQUFDQSxJQUFJQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtBQUNwQkEsQ0FBQ0E7QUFGZSxXQUFHLEdBQUgsR0FFZixDQUFBIiwiZmlsZSI6InNyYy9zcXVhc2guanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgU3luYXB0aWMgPSByZXF1aXJlKCcuL3N5bmFwdGljJyk7XG5cbi8vIHNxdWFzaGluZyBmdW5jdGlvbnNcblxuZXhwb3J0IGZ1bmN0aW9uIExPR0lTVElDKHg6IG51bWJlciwgZGVyaXZhdGU/OiBib29sZWFuKSB7XG5cdGlmIChkZXJpdmF0ZSkge1xuXHRcdHZhciBmeCA9IExPR0lTVElDKHgpO1xuXHRcdHJldHVybiBmeCAqICgxIC0gZngpO1xuXHR9XG5cdHJldHVybiAxIC8gKDEgKyBNYXRoLmV4cCgteCkpO1xuXG59XG5cbmV4cG9ydCBmdW5jdGlvbiBUQU5IKHg6IG51bWJlciwgZGVyaXZhdGU/OiBib29sZWFuKSB7XG5cdGlmIChkZXJpdmF0ZSlcblx0XHRyZXR1cm4gMSAtIE1hdGgucG93KFRBTkgoeCksIDIpO1xuXHR2YXIgZVAgPSBNYXRoLmV4cCh4KTtcblx0dmFyIGVOID0gMSAvIGVQO1xuXHRyZXR1cm4gKGVQIC0gZU4pIC8gKGVQICsgZU4pO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gSURFTlRJVFkoeDogbnVtYmVyLCBkZXJpdmF0ZT86IGJvb2xlYW4pIHtcblx0cmV0dXJuIGRlcml2YXRlID8gMSA6IHg7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBITElNKHg6IG51bWJlciwgZGVyaXZhdGU/OiBib29sZWFuKSB7XG5cdHJldHVybiBkZXJpdmF0ZSA/IDEgOiArKHggPiAwKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIFNPRlRQTFVTKHg6IG51bWJlciwgZGVyaXZhdGU/OiBib29sZWFuKSB7XG5cdGlmIChkZXJpdmF0ZSlcblx0XHRyZXR1cm4gMSAtIDEgLyAoMSArIE1hdGguZXhwKHgpKTtcblx0cmV0dXJuIE1hdGgubG9nKDEgKyBNYXRoLmV4cCh4KSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBFWFAoeDogbnVtYmVyLCBkZXJpdmF0ZT86IGJvb2xlYW4pIHtcblx0cmV0dXJuIE1hdGguZXhwKHgpO1xufSAiXSwic291cmNlUm9vdCI6Ii9zb3VyY2UvIn0=