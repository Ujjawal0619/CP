-----------------   Axios ----------------------------

// SETTING TORKEN TO GLOBAL HEADERS
(function() {
     String token = store.getState().session.token;
     if (token) {
         axios.defaults.headers.common['Authorization'] = token;
     } else {
         axios.defaults.headers.common['Authorization'] = null;
         /*if setting null does not remove `Authorization` header then try     
           delete axios.defaults.headers.common['Authorization'];
         */
     }
})();

// using intersectper
// Add a request interceptor
axios.interceptors.request.use(function (config) {
    const token = store.getState().session.token;
    config.headers.Authorization =  token;

    return config;
});

Note: `Bearer ${token}` may need to add as requirenment



How to join two JavaScript Objects, without using JQUERY [duplicate]
--------------------------------------------------------------------
https://stackoverflow.com/questions/21450060/how-to-join-two-javascript-objects-without-using-jquery



Join two array in Js
--------------------
allResult.concat(allPayment);



Date formate
-------------
var options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
var today  = new Date();

console.log(today.toLocaleDateString("en-US")); // 9/17/2016
console.log(today.toLocaleDateString("en-US", options)); // Saturday, September 17, 2016
console.log(today.toLocaleDateString("hi-IN", options)); // शनिवार, 17 सितंबर 2016

more: https://stackoverflow.com/a/34015511