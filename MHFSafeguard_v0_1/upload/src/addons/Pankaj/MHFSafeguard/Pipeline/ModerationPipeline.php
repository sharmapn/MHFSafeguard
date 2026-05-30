<?php

namespace Pankaj\MHFSafeguard\Pipeline;

use Pankaj\MHFSafeguard\Content\ContentContext;
use Pankaj\MHFSafeguard\Gateway\ClassifierGateway;
use Pankaj\MHFSafeguard\Repository\SafeguardScan;

class ModerationPipeline
{
    protected $normaliser;
    protected $payloadBuilder;
    protected $gateway;
    protected $interpreter;
    protected $decision;
    protected $repository;

    public function __construct()
    {
        $this->normaliser = new TextNormaliser();
        $this->payloadBuilder = new PayloadBuilder();
        $this->gateway = new ClassifierGateway();
        $this->interpreter = new ResponseInterpreter();
        $this->decision = new PolicyDecision();
        $this->repository = new SafeguardScan();
    }

    public function scan(ContentContext $context): array
    {
        $cleanMessage = $this->normaliser->normalise($context->getMessage());

        if ($cleanMessage === '')
        {
            return [
                'clean_message' => '',
                'result' => [
                    'api_success' => true,
                    'risk_level' => 'none',
                    'recommended_action' => 'allow',
                    'highest_label' => '',
                    'highest_score' => 0,
                    'flagged_parts' => [],
                    'raw_response' => ''
                ],
                'final_action' => 'allow'
            ];
        }

        $payload = $this->payloadBuilder->build($context, $cleanMessage);
        $gatewayResponse = $this->gateway->classify($payload);
        $result = $this->interpreter->interpret($gatewayResponse);
        $finalAction = $this->decision->decide($result);

        return [
            'clean_message' => $cleanMessage,
            'result' => $result,
            'final_action' => $finalAction
        ];
    }

    public function record(ContentContext $context, string $cleanMessage, array $result, string $finalAction): void
    {
        $this->repository->log($context, $cleanMessage, $result, $finalAction);
    }
}
