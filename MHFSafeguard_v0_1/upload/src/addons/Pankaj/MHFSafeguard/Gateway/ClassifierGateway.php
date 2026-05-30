<?php

namespace Pankaj\MHFSafeguard\Gateway;

class ClassifierGateway
{
    public function classify(array $payload): array
    {
        $options = \XF::options();
        $apiUrl = trim((string)$options->mhfsApiUrl);
        $apiKey = trim((string)$options->mhfsApiKey);
        $timeout = (int)$options->mhfsTimeout;

        if ($timeout <= 0)
        {
            $timeout = 8;
        }

        if ($apiUrl === '')
        {
            return [
                'ok' => false,
                'status' => 0,
                'error' => 'Classifier API URL is not configured.',
                'raw' => '',
                'data' => []
            ];
        }

        $headers = [
            'Content-Type' => 'application/json',
            'Accept' => 'application/json',
            'User-Agent' => 'Pankaj-MHFSafeguard-XenForo/0.1'
        ];

        if ($apiKey !== '')
        {
            $headers['Authorization'] = 'Bearer ' . $apiKey;
        }

        try
        {
            $client = \XF::app()->http()->client();
            $response = $client->post($apiUrl, [
                'headers' => $headers,
                'body' => json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES),
                'timeout' => $timeout,
                'http_errors' => false
            ]);

            $status = (int)$response->getStatusCode();
            $raw = (string)$response->getBody()->getContents();
            $data = json_decode($raw, true);

            if (!is_array($data))
            {
                $data = [];
            }

            return [
                'ok' => ($status >= 200 && $status < 300),
                'status' => $status,
                'error' => ($status >= 200 && $status < 300) ? '' : ('HTTP ' . $status),
                'raw' => $raw,
                'data' => $data
            ];
        }
        catch (\Throwable $e)
        {
            return [
                'ok' => false,
                'status' => 0,
                'error' => $e->getMessage(),
                'raw' => '',
                'data' => []
            ];
        }
    }
}
