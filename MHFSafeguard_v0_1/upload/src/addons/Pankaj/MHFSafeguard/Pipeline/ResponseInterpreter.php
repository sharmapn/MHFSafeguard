<?php

namespace Pankaj\MHFSafeguard\Pipeline;

class ResponseInterpreter
{
    public function interpret(array $gatewayResponse): array
    {
        $data = $gatewayResponse['data'] ?? [];
        $score = $this->normaliseScore($data['highest_score'] ?? 0);

        return [
            'api_success' => !empty($gatewayResponse['ok']),
            'api_status_code' => (int)($gatewayResponse['status'] ?? 0),
            'api_error' => (string)($gatewayResponse['error'] ?? ''),
            'risk_level' => (string)($data['risk_level'] ?? 'none'),
            'recommended_action' => (string)($data['recommended_action'] ?? $data['action'] ?? 'allow'),
            'highest_label' => (string)($data['highest_label'] ?? ''),
            'highest_score' => $score,
            'flagged_parts' => $this->extractFlaggedParts($data),
            'raw_response' => (string)($gatewayResponse['raw'] ?? '')
        ];
    }

    protected function normaliseScore($score): float
    {
        $score = (float)$score;

        if ($score > 0 && $score <= 1)
        {
            $score = $score * 100;
        }

        if ($score < 0) { $score = 0; }
        if ($score > 100) { $score = 100; }

        return $score;
    }

    protected function extractFlaggedParts(array $data): array
    {
        foreach (['flagged_parts', 'flagged_spans', 'spans', 'sentence_results', 'sentences'] as $key)
        {
            if (isset($data[$key]) && is_array($data[$key]))
            {
                return $data[$key];
            }
        }

        return [];
    }
}
