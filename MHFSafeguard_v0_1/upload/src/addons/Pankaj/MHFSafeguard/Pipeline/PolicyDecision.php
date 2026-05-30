<?php

namespace Pankaj\MHFSafeguard\Pipeline;

class PolicyDecision
{
    public function decide(array $result): string
    {
        $options = \XF::options();
        $mode = strtolower(trim((string)$options->mhfsActionMode));
        $moderateThreshold = (float)$options->mhfsModerateThreshold;
        $reviseThreshold = (float)$options->mhfsReviseThreshold;
        $failOpen = (bool)$options->mhfsFailOpen;

        if ($mode === '')
        {
            $mode = 'moderate';
        }

        if (empty($result['api_success']))
        {
            return $failOpen ? 'allow' : 'moderate';
        }

        if ($mode === 'log')
        {
            return 'allow';
        }

        $recommended = strtolower((string)($result['recommended_action'] ?? 'allow'));
        $score = (float)($result['highest_score'] ?? 0);

        if (in_array($recommended, ['block', 'reject', 'revise', 'revision'], true))
        {
            return ($mode === 'revise') ? 'revise' : 'moderate';
        }

        if (in_array($recommended, ['moderate', 'flag', 'review'], true))
        {
            return 'moderate';
        }

        if ($mode === 'revise' && $reviseThreshold > 0 && $score >= $reviseThreshold)
        {
            return 'revise';
        }

        if ($moderateThreshold > 0 && $score >= $moderateThreshold)
        {
            return 'moderate';
        }

        return 'allow';
    }
}
