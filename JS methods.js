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
-----------------------------------------------------------