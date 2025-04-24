const path = require('path');

module.exports = {
  webpack: function(config, env) {
    // Add the fallback for 'http' module
    config.resolve.fallback = {
      ...config.resolve.fallback,
      http: require.resolve('stream-http'),
    };
    return config;
  },
};
